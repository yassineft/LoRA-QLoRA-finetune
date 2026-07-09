from load import load_model, generate_text_stream


def chat(adapter_dir="./output"):
    try:
        model, tokenizer = load_model(adapter_dir=adapter_dir)
    except KeyboardInterrupt:
        print("\nInterrupted while loading the model. Exiting.")
        return

    print("Loaded QLoRA model. Type 'exit' to quit.")

    while True:
        try:
            prompt = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        print("Assistant: ", end="", flush=True)
        for chunk in generate_text_stream(model, tokenizer, prompt):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    chat()
