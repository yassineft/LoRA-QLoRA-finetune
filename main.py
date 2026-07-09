import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import prepare_model_for_kbit_training, get_peft_model

# local modules
from tokenzer import tokenizer, model_name
from data_utils import train_dataset, validation_dataset
from config import peft_config


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    tokenized["labels"] = [
        [
            -100 if token_id == tokenizer.pad_token_id else token_id
            for token_id in input_ids
        ]
        for input_ids in tokenized["input_ids"]
    ]
    return tokenized


train_dataset = train_dataset.map(tokenize_function, batched=True)
validation_dataset = validation_dataset.map(tokenize_function, batched=True)


# BitsAndBytes 4-bit config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


def supports_cuda_runtime():
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (7, 5)
    except Exception:
        return False

use_cuda = supports_cuda_runtime()
if not use_cuda:
    print("CUDA is unavailable or unsupported by the installed PyTorch build. Forcing CPU-only model load.")

device_map = "auto" if use_cuda else "cpu"
torch_dtype = torch.float16 if use_cuda else torch.float32

# Load the base model in 4-bit if possible
print("Loading base model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype if use_cuda else None,
        trust_remote_code=True,
    )
except ImportError as error:
    print("bitsandbytes 4-bit quantization is unavailable:", error)
    print("Falling back to standard model load without 4-bit quantization.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

# Enable gradient checkpointing and disable cache for training
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Prepare for k-bit training and apply LoRA adapters
print("Loading QLoRA adapter...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    fp16=use_cuda,
    use_cpu=not use_cuda,
    report_to="tensorboard",
)


def count_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return trainable, total


trainable, total = count_trainable_parameters(model)
print(f"Trainable params: {trainable} / {total} ({100 * trainable/total:.2f}%)")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=default_data_collator,
)


if __name__ == "__main__":
    trainer.train()
    trainer.save_model(training_args.output_dir)
