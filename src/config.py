"""MicroMixer-1 model configuration dataclasses."""

from dataclasses import dataclass


@dataclass
class MicroMixerConfig:
    """Configuration for MicroMixer-1 MLP-Mixer language model.

    Attributes:
        vocab_size: Vocabulary size for byte-level tokenization (default: 256).
        max_seq_len: Maximum sequence length the model can process.
        hidden_dim: Hidden dimension size for token embeddings and mixer layers.
        token_mlp_dim: Inner dimension of the token-mixing MLP (projects across tokens).
        channel_mlp_dim: Inner dimension of the channel-mixing MLP (projects across channels).
        num_layers: Number of mixer layers in the model.
        dropout: Dropout probability for regularization (default: 0.1).
    """

    max_seq_len: int
    hidden_dim: int
    token_mlp_dim: int
    channel_mlp_dim: int
    num_layers: int
    vocab_size: int = 256
    dropout: float = 0.1


MICRO_100K = MicroMixerConfig(
    max_seq_len=128,
    hidden_dim=64,
    token_mlp_dim=64,
    channel_mlp_dim=128,
    num_layers=2,
)

MICRO_300K = MicroMixerConfig(
    max_seq_len=128,
    hidden_dim=128,
    token_mlp_dim=128,
    channel_mlp_dim=256,
    num_layers=4,
)

MICRO_500K = MicroMixerConfig(
    max_seq_len=192,
    hidden_dim=128,
    token_mlp_dim=128,
    channel_mlp_dim=256,
    num_layers=4,
)

MICRO_1M = MicroMixerConfig(
    max_seq_len=192,
    hidden_dim=192,
    token_mlp_dim=192,
    channel_mlp_dim=384,
    num_layers=4,
)


def get_config(size: str) -> MicroMixerConfig:
    """Get a preset MicroMixer configuration by size.

    Args:
        size: Model size string. One of "100k", "300k", "500k", "1M".

    Returns:
        The corresponding MicroMixerConfig preset.

    Raises:
        ValueError: If size is not a recognized preset.
    """
    size_lower = size.lower()
    if size_lower == "100k":
        return MICRO_100K
    elif size_lower == "300k":
        return MICRO_300K
    elif size_lower == "500k":
        return MICRO_500K
    elif size_lower == "1m":
        return MICRO_1M
    else:
        raise ValueError(f"Unknown model size: {size}. Valid options: '100k', '300k', '500k', '1M'")
