import unittest

from offline.asr_models import (
    TransformersWhisperTranscriber,
    _initialize_cuda_memory_measurement,
    load_asr_model_config,
    normalize_whisper_language,
)


class AsrModelRegistryTests(unittest.TestCase):
    def test_cuda_memory_measurement_uses_initialized_current_device(self):
        calls = []

        class FakeCuda:
            @staticmethod
            def current_device():
                calls.append(("current_device",))
                return 1

            @staticmethod
            def set_device(device):
                calls.append(("set_device", device))

            @staticmethod
            def empty_cache():
                calls.append(("empty_cache",))

            @staticmethod
            def reset_peak_memory_stats(*args):
                calls.append(("reset_peak_memory_stats", *args))

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def device(value):
                return f"device:{value}"

            @staticmethod
            def empty(size, *, device):
                calls.append(("empty", size, device))
                return object()

        device_index, cuda_device = _initialize_cuda_memory_measurement(FakeTorch())

        self.assertEqual(device_index, 1)
        self.assertEqual(cuda_device, "device:cuda:1")
        self.assertIn(("empty", 1, "device:cuda:1"), calls)
        self.assertIn(("reset_peak_memory_stats",), calls)
        self.assertNotIn(("reset_peak_memory_stats", 0), calls)

    def test_pins_both_dev_gate_candidates_to_immutable_revisions(self):
        whisper = load_asr_model_config("whisper_large_v3")
        pho = load_asr_model_config("phowhisper_large")
        self.assertEqual(len(whisper["revision"]), 40)
        self.assertEqual(whisper["weight_format"], "safetensors")
        self.assertEqual(len(whisper["weight_sha256"]), 64)
        self.assertEqual(pho["weight_format"], "pytorch_bin_weights_only")
        self.assertIsNone(pho["weight_sha256"])
        self.assertFalse(pho["production_allowed"])

    def test_models_remain_fail_closed_for_production_before_gate(self):
        with self.assertRaisesRegex(RuntimeError, "not approved for production"):
            TransformersWhisperTranscriber(
                model_key="whisper_large_v3", purpose="production"
            )
        with self.assertRaisesRegex(RuntimeError, "not approved for production"):
            TransformersWhisperTranscriber(
                model_key="phowhisper_large", purpose="production"
            )

    def test_normalizes_only_contract_languages(self):
        self.assertEqual(normalize_whisper_language("<|vi|>"), "vi")
        self.assertEqual(normalize_whisper_language("English"), "en")
        with self.assertRaisesRegex(RuntimeError, "unsupported language"):
            normalize_whisper_language("fr")


if __name__ == "__main__":
    unittest.main()
