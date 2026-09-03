param(
    [string]$VenvPath = ".venv-offline",
    [string]$PythonCommand = "py",
    [switch]$DevOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedVenv = [IO.Path]::GetFullPath((Join-Path $repoRoot $VenvPath))
if (-not $resolvedVenv.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw "VenvPath must stay inside the repository: $resolvedVenv"
}

if ($PythonCommand -eq "py") {
    & py -3.11 -m venv $resolvedVenv
} else {
    & $PythonCommand -m venv $resolvedVenv
}

$python = Join-Path $resolvedVenv "Scripts\python.exe"
& $python -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"

$profile = "offline-local"
$requirements = Join-Path $repoRoot "requirements\offline-local.txt"
if ($DevOnly) {
    $profile = "dev"
    $requirements = Join-Path $repoRoot "requirements\dev.txt"
}

& $python -m pip install -r $requirements
Push-Location $repoRoot
try {
    & $python -m scripts.environment_doctor --profile $profile
    & $python -m compileall -q offline shared scripts tests
    & $python -m unittest discover -s tests -q
} finally {
    Pop-Location
}
