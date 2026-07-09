import unittest
from unittest.mock import patch

import torch

import chat
from load import generate_text, generate_text_stream


class ChatTests(unittest.TestCase):
    def test_chat_exits_gracefully_on_keyboard_interrupt(self):
        with patch("chat.load_model", return_value=("model", "tokenizer")), patch(
            "builtins.input", side_effect=KeyboardInterrupt
        ), patch("builtins.print") as mock_print:
            chat.chat(adapter_dir="./output")

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("Goodbye!", output)

    def test_generate_text_streams_incremental_chunks(self):
        class DummyTokenizer:
            eos_token_id = 0

            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": torch.tensor([[1]])}

            def decode(self, output, skip_special_tokens=True):
                return "streaming ok"

        class DummyModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return torch.tensor([[1]])

        model = DummyModel()
        tokenizer = DummyTokenizer()

        chunks = list(generate_text_stream(model, tokenizer, "hello", max_new_tokens=6))

        self.assertEqual(chunks, ["streaming ok"])

    def test_generate_text_falls_back_to_cpu_on_cuda_error(self):
        class DummyTokenizer:
            eos_token_id = 0

            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": torch.tensor([[1]])}

            def decode(self, output, skip_special_tokens=True):
                return "fallback ok"

        class DummyModel:
            def __init__(self):
                self.device = torch.device("cpu")
                self.generate_calls = []

            def to(self, device):
                self.device = torch.device(device)
                return self

            def generate(self, **kwargs):
                input_ids = kwargs.get("input_ids")
                self.generate_calls.append(getattr(input_ids, "device", torch.device("cpu")))
                if input_ids is not None and input_ids.device.type == "cuda":
                    raise torch.AcceleratorError("CUDA error: no kernel image is available")
                return torch.tensor([[1]])

        model = DummyModel()
        tokenizer = DummyTokenizer()

        with patch("load.supports_cuda_runtime", return_value=True):
            response = generate_text(model, tokenizer, "hello")

        self.assertEqual(response, "fallback ok")
        self.assertEqual([device.type for device in model.generate_calls], ["cuda", "cpu"])


if __name__ == "__main__":
    unittest.main()
