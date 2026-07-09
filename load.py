from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def _version_ge(v1: str, v2: str) -> bool:
    def to_tuple(v: str):
        parts = []
        for p in v.split("."):
            try:
                parts.append(int(p))
            except Exception:
                # stop at first non-int part
                break
        return tuple(parts)

    return to_tuple(v1) >= to_tuple(v2)


def supports_cuda_runtime() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (7, 5)
    except Exception:
        return False

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_DIR = "./output"


def resolve_adapter_dir(adapter_dir=DEFAULT_ADAPTER_DIR):
    base_dir = Path(adapter_dir)
    if not base_dir.is_absolute():
        base_dir = (Path(__file__).resolve().parent / base_dir).resolve()

    if base_dir.is_file():
        return base_dir.parent

    if (base_dir / "adapter_config.json").exists():
        return base_dir

    checkpoint_dirs = sorted(
        [p for p in base_dir.glob("checkpoint-*") if (p / "adapter_config.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1]

    for match in base_dir.rglob("adapter_config.json"):
        if match.is_file():
            return match.parent

    workspace_dir = Path(__file__).resolve().parent
    for match in workspace_dir.rglob("adapter_config.json"):
        if match.is_file():
            return match.parent

    return base_dir


def load_model(adapter_dir=DEFAULT_ADAPTER_DIR):
    input_path = Path(adapter_dir)
    if not input_path.is_absolute():
        input_path = (Path(__file__).resolve().parent / input_path).resolve()

    adapter_path = resolve_adapter_dir(adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        if input_path.is_file():
            raise FileNotFoundError(
                f"The provided path '{input_path}' is a file, not a LoRA adapter directory. "
                "Use a folder that contains adapter_config.json and adapter_model.safetensors."
            )
        raise FileNotFoundError(
            f"No LoRA adapter was found in '{adapter_path}'. "
            "Train the model first or pass a folder that contains adapter_config.json."
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Decide on runtime/device
    use_cuda = supports_cuda_runtime()
    device_map = "auto" if use_cuda else "cpu"
    torch_dtype = torch.float16 if use_cuda else torch.float32

    # Try to enable bitsandbytes 4-bit quantization if available and recent enough
    bnb_config = None
    try:
        import bitsandbytes as bnb  # type: ignore

        if _version_ge(getattr(bnb, "__version__", "0"), "0.46.1") and use_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            # bitsandbytes is present but either too old or CUDA/runtime unsupported
            bnb_config = None
    except Exception:
        bnb_config = None

    # Load model, falling back to non-quantized load if necessary
    if bnb_config is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception as e:
            print("bitsandbytes 4-bit quantization load failed:", e)
            print("Falling back to standard model load.")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def _run_generate(model, tokenizer, inputs, max_new_tokens, use_cuda):
    with torch.no_grad():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


def generate_text(model, tokenizer, prompt, max_new_tokens=200):
    use_cuda = supports_cuda_runtime()
    device = "cuda" if use_cuda else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        outputs = _run_generate(model, tokenizer, inputs, max_new_tokens, use_cuda)
    except Exception as exc:
        if not use_cuda:
            raise

        if not isinstance(exc, (RuntimeError, torch.AcceleratorError)) and "cuda" not in str(exc).lower():
            raise

        print(f"CUDA generation failed ({exc}). Falling back to CPU.")
        try:
            model = model.to("cpu")
        except Exception:
            pass

        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        outputs = _run_generate(model, tokenizer, inputs, max_new_tokens, False)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_text_stream(model, tokenizer, prompt, max_new_tokens=200):
    use_cuda = supports_cuda_runtime()
    device = "cuda" if use_cuda else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    except Exception as exc:
        if not use_cuda:
            raise

        if not isinstance(exc, (RuntimeError, torch.AcceleratorError)) and "cuda" not in str(exc).lower():
            raise

        print(f"CUDA generation failed ({exc}). Falling back to CPU.")
        try:
            model = model.to("cpu")
        except Exception:
            pass

        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

    if hasattr(generated, "sequences"):
        sequences = generated.sequences
        generated_tokens = sequences[0][inputs["input_ids"].shape[1]:]
        full_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    else:
        generated_tokens = generated[0][inputs["input_ids"].shape[1]:]
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    if generated_tokens.numel() == 0:
        yield full_text
        return

    for token_id in generated_tokens.tolist():
        chunk = tokenizer.decode([token_id], skip_special_tokens=True)
        if chunk:
            yield chunk


if __name__ == "__main__":
    model, tokenizer = load_model()
    prompt = "Explain computational thinking in simple words."
    print(generate_text(model, tokenizer, prompt))
