"""MicroMixer-1 training script.

Usage:
    python train.py --model 100k --epochs 10 --batch-size 32
    python train.py --model 1M --epochs 5 --max-samples 1000
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.model import (
    MicroMixer, MicroMixerV1, MicroMixerV2,
    micromixer_100k, micromixer_300k, micromixer_500k, micromixer_1m,
    micromixer_v2_100k, micromixer_v2_300k, micromixer_v2_500k, micromixer_v2_1m,
    count_parameters,
)
from src.tokenizer import ByteTokenizer
from src.data import load_combined_dataset, create_dataloader
from src.trainer import Trainer


def get_model_config(size: str, version: str = "v1"):
    v1_configs = {
        "100k": micromixer_100k,
        "300k": micromixer_300k,
        "500k": micromixer_500k,
        "1M": micromixer_1m,
    }
    v2_configs = {
        "100k": micromixer_v2_100k,
        "300k": micromixer_v2_300k,
        "500k": micromixer_v2_500k,
        "1M": micromixer_v2_1m,
    }
    
    if version == "v1":
        configs = v1_configs
    elif version == "v2":
        configs = v2_configs
    else:
        raise ValueError(f"Unknown version: {version}. Choose 'v1' or 'v2'")
    
    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from: {list(configs.keys())}")
    return configs[size]()


def main():
    parser = argparse.ArgumentParser(description="Train MicroMixer-1 language model")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="100k", 
                        choices=["100k", "300k", "500k", "1M"],
                        help="Model size to train")
    parser.add_argument("--version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="Model version: v1 (original) or v2 (SGU/HyperMixing/RoPE)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    
    # Data arguments
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (for debugging)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Logging arguments
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 60)
    print("MicroMixer-1 Training")
    print("=" * 60)
    
    # Load model config
    config = get_model_config(args.model, args.version)
    print(f"\nModel: {args.model} ({args.version})")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  token_mlp_dim: {config.token_mlp_dim}")
    print(f"  channel_mlp_dim: {config.channel_mlp_dim}")
    print(f"  num_layers: {config.num_layers}")
    
    if args.version == "v1":
        model = MicroMixerV1(config)
    else:
        model = MicroMixerV2(config)
    
    param_count = count_parameters(model)
    print(f"  Parameters: {param_count:,}")
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Load dataset
    print(f"\nLoading datasets...")
    print(f"  max_samples per dataset: {args.max_samples or 'all'}")
    
    texts = load_combined_dataset(
        max_daily_dialog=args.max_samples,
        max_tiny_stories=args.max_samples,
    )
    print(f"  Total samples: {len(texts)}")
    
    # Split train/val
    val_size = int(len(texts) * args.val_split)
    train_texts = texts[val_size:]
    val_texts = texts[:val_size]
    print(f"  Train samples: {len(train_texts)}")
    print(f"  Val samples: {len(val_texts)}")
    
    # Create dataloaders
    max_length = min(args.max_length, config.max_seq_len)
    
    train_loader = create_dataloader(
        train_texts, tokenizer, max_length, args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_texts, tokenizer, max_length, args.batch_size, shuffle=False
    ) if val_texts else None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        grad_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
