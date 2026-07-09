# LoRA Fine-Tuning for Qwen

This project fine-tunes a Qwen instruct model with LoRA/PEFT and provides a simple chat interface for interacting with the adapted model.

## What this project does

- Trains a LoRA adapter on your dataset using main.py
- Loads the base model plus adapter with load.py
- Starts a conversational chat loop with chat.py

## Requirements

- Python 3.10+ recommended
- A CUDA-capable GPU is strongly recommended for faster training and inference
- Install dependencies:

```powershell
pip install -r requirements.txt
```

If you are using PowerShell and activation is blocked, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv lora_env
.\lora_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the model requires authentication, log in to Hugging Face:

```powershell
huggingface-cli login
```

## Fine-tune the model

1. Prepare your training data in data_utils.py and the related dataset files such as dataset.json.
2. Start training with:

```powershell
python main.py
```

Training outputs are written to the output folder. The script saves checkpoints and the final adapter there.

## Start chatting

After training has produced an adapter, start the chat interface with:

```powershell
python chat.py
```

You can then type prompts in the terminal. Type exit or quit to stop the session.

## Notes

- The chat script uses the adapter in output by default.
- If you want to use a different adapter directory, update the adapter_dir value in chat.py.
- If CUDA is unavailable, the scripts will fall back to CPU loading, though training and inference will be slower.

## Project files

- main.py - LoRA training loop
- chat.py - Interactive chat interface
- load.py - Model and adapter loading logic
- config.py - LoRA configuration
- tokenzer.py - Tokenizer and model settings
- data_utils.py - Dataset preparation
