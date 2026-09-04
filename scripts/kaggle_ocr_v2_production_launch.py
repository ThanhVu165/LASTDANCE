import json, os, signal, subprocess, sys, threading, time
from contextlib import contextmanager
from pathlib import Path
from kaggle_secrets import UserSecretsClient

@contextmanager
def progress(stage, interval=20):
    started = time.monotonic()
    stop = threading.Event()
    print(f'[START] {stage}', flush=True)
    def tick():
        while not stop.wait(interval):
            print(f'[WAIT] {stage} | elapsed={time.monotonic() - started:.0f}s | '
                  'still waiting; progress/ETA unavailable unless reported below', flush=True)
    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as exc:
        print(f'[STOP] {stage} | {type(exc).__name__} | elapsed={time.monotonic() - started:.0f}s', flush=True)
        raise
    else:
        print(f'[DONE] {stage} | elapsed={time.monotonic() - started:.0f}s', flush=True)
    finally:
        stop.set()
        thread.join()

def run_logged(command, stage, env):
    # Forward child stdout/stderr into notebook output; inherited OS stdout may be invisible.
    with progress(stage):
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
            bufsize=1, start_new_session=(os.name != 'nt'))
        try:
            for line in process.stdout:
                secret = env.get('HF_TOKEN')
                print(line.replace(secret, '[REDACTED]') if secret else line, end='', flush=True)
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except BaseException:
            # Stop the Linux process group too, including pip/model subprocesses.
            if os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise
        finally:
            process.stdout.close()

root = Path(os.environ.get('AIC_DATA', '/kaggle/working')) / 'ocr-v2-production'
code_dir = root / 'runtime'
with progress('1/5 Write runtime files'):
    code_dir.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        (code_dir / name).write_text(source, encoding='utf-8')
if ACTION not in ('plan', 'run'):
    raise ValueError('ACTION must be plan/run')
with progress('2/5 Locate worker plan'):
    plan_path = WORKER_PLAN
    if ACTION == 'run' and not plan_path:
        matches = list(Path(INPUT_ROOT).rglob('ocr-v2-worker-plan.json'))
        if len(matches) != 1:
            raise ValueError('Attach exactly one worker plan or set WORKER_PLAN')
        plan_path = str(matches[0])
    if ACTION == 'run':
        if not Path(plan_path).is_file():
            raise ValueError('WORKER_PLAN does not exist: ' + plan_path)
        print('WORKER_PLAN', plan_path, flush=True)
with progress('3/5 Read Kaggle Secret HF_TOKEN'):
    token = UserSecretsClient().get_secret('HF_TOKEN')
child_env = {**os.environ, 'HF_TOKEN': token, 'CUDA_VISIBLE_DEVICES': '0',
             'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
if ACTION == 'run' and RUN_SETUP:
    run_logged([sys.executable, '-u', str(code_dir / 'ocr_v2_environment.py'), str(root / 'env-cache')],
               '4/5 Environment setup and GPU probes', child_env)
else:
    print('[SKIP] 4/5 Environment setup', flush=True)
config = {'repo': HF_REPO_ID, 'input_revision': INPUT_REVISION,
          'catalog_path': CATALOG_PATH, 'input_root': INPUT_ROOT,
          'output': str(root), 'plan': plan_path, 'keyframes': KEYFRAMES_ROOT,
          'worker': WORKER_SLOT, 'mode': RUN_MODE, 'interrupt_after': INTERRUPT_AFTER_MINIBATCHES,
          'approved_canary_sha256': APPROVED_CANARY_SHA256}
config_path = root / 'worker-config.json'
config_path.write_text(json.dumps(config), encoding='utf-8')
run_logged([sys.executable, '-u', str(code_dir / 'kaggle_ocr_v2_production_runtime.py'), ACTION, str(config_path)],
           f'5/5 OCR {ACTION} | worker={WORKER_SLOT} | mode={RUN_MODE}', child_env)
print('OUTPUT_ROOT', root, flush=True)
