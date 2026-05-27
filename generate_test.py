import torch
from src.model import MicroMixerV2, MicroMixerV2Config, count_parameters
from src.tokenizer import ByteTokenizer


def get_v2_hyper_configs():
    return {
        "100k": lambda: MicroMixerV2Config(
            max_seq_len=64, hidden_dim=84, channel_mlp_dim=128,
            num_layers=3, dropout=0.1, use_hyper=True,
        ),
        "300k": lambda: MicroMixerV2Config(
            max_seq_len=128, hidden_dim=128, channel_mlp_dim=288,
            num_layers=3, dropout=0.1, use_hyper=True,
        ),
        "500k": lambda: MicroMixerV2Config(
            max_seq_len=128, hidden_dim=176, channel_mlp_dim=384,
            num_layers=3, dropout=0.1, use_hyper=True,
        ),
        "1M": lambda: MicroMixerV2Config(
            max_seq_len=256, hidden_dim=224, channel_mlp_dim=576,
            num_layers=3, dropout=0.1, use_hyper=True,
        ),
    }


def load_model(checkpoint_path: str, config: MicroMixerV2Config, device: str):
    model = MicroMixerV2(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt: str, device: str,
                  max_new_tokens: int = 64, temperature: float = 0.8, top_k: int = 40):
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    generated = output_ids[0].cpu().tolist()
    return tokenizer.decode(generated)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 70)

    tokenizer = ByteTokenizer()
    configs = get_v2_hyper_configs()

    prompts = [
        "Hello",
        "The weather is",
        "I want to",
        "Once upon a time",
    ]

    for size_name in ["100k", "300k", "500k", "1M"]:
        config_factory = configs[size_name]
        config = config_factory()

        print(f"\n{'#' * 70}")
        print(f"# Model: v2_hyper_{size_name}")
        print(f"# Params: {count_parameters(MicroMixerV2(config)):,}")
        print(f"# max_seq_len={config.max_seq_len}, hidden_dim={config.hidden_dim}")
        print(f"{'#' * 70}")

        for epoch in range(5):
            ckpt_path = f"checkpoints/v2_hyper_{size_name}/epoch_{epoch}.pt"
            model = load_model(ckpt_path, config, device)

            print(f"\n  --- Epoch {epoch} ---")

            for prompt in prompts:
                max_prompt_len = config.max_seq_len // 2
                encoded = tokenizer.encode(prompt)
                if len(encoded) > max_prompt_len:
                    encoded = encoded[:max_prompt_len]
                    prompt_trimmed = tokenizer.decode(encoded)
                else:
                    prompt_trimmed = prompt

                result = generate_text(
                    model, tokenizer, prompt_trimmed, device,
                    max_new_tokens=48, temperature=0.8, top_k=40,
                )
                result_display = result.replace("\n", "↵")
                print(f"    [{prompt:>20s}] → {result_display}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
