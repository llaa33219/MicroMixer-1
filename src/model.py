import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CausalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features != out_features:
            raise ValueError(
                f"CausalLinear requires in_features == out_features, got "
                f"{in_features} and {out_features}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.weight.copy_(torch.tril(self.weight))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-1)
        weight = torch.tril(self.weight)[:seq_len, :seq_len]
        bias = self.bias[:seq_len] if self.bias is not None else None
        return F.linear(x, weight, bias)


class CausalMlpBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        if dim != hidden_dim:
            raise ValueError(
                f"CausalMlpBlock requires dim == hidden_dim for causal masking, "
                f"got {dim} and {hidden_dim}"
            )
        self.fc1 = CausalLinear(dim, hidden_dim)
        self.fc2 = CausalLinear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CausalMixerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if token_mlp_dim != max_seq_len:
            raise ValueError(
                f"token_mlp_dim ({token_mlp_dim}) must equal max_seq_len "
                f"({max_seq_len}) for causal token mixing"
            )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.token_mlp = CausalMlpBlock(max_seq_len, token_mlp_dim, dropout)
        self.channel_mlp = MlpBlock(hidden_dim, channel_mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.token_mlp(x)
        x = x.transpose(1, 2)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.channel_mlp(x)
        x = x + residual

        return x


class MicroMixerV1(nn.Module):
    def __init__(self, config: "MicroMixerConfig"):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.hidden_dim = config.hidden_dim

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.mixer_layers = nn.ModuleList([
            CausalMixerLayer(
                config.hidden_dim,
                config.token_mlp_dim,
                config.channel_mlp_dim,
                config.max_seq_len,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if getattr(config, "tie_weights", True):
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, CausalLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, S = input_ids.shape

        if S > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]
            S = self.max_seq_len
            if targets is not None:
                targets = targets[:, -self.max_seq_len :]

        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(S, device=input_ids.device)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        for layer in self.mixer_layers:
            x = layer(x)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )
            return logits, loss

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        for _ in range(max_new_tokens):
            logits = self(input_ids)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


MicroMixer = MicroMixerV1


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class MicroMixerConfig:
    vocab_size: int = 256
    max_seq_len: int = 128
    hidden_dim: int = 128
    token_mlp_dim: int = 128
    channel_mlp_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    tie_weights: bool = True


def micromixer_100k() -> MicroMixerConfig:
    return MicroMixerConfig(
        max_seq_len=64,
        hidden_dim=64,
        token_mlp_dim=64,
        channel_mlp_dim=128,
        num_layers=3,
        dropout=0.1,
    )


def micromixer_300k() -> MicroMixerConfig:
    return MicroMixerConfig(
        max_seq_len=128,
        hidden_dim=160,
        token_mlp_dim=128,
        channel_mlp_dim=256,
        num_layers=2,
        dropout=0.1,
    )


def micromixer_500k() -> MicroMixerConfig:
    return MicroMixerConfig(
        max_seq_len=128,
        hidden_dim=128,
        token_mlp_dim=128,
        channel_mlp_dim=448,
        num_layers=3,
        dropout=0.1,
    )


def micromixer_1m() -> MicroMixerConfig:
    return MicroMixerConfig(
        max_seq_len=256,
        hidden_dim=288,
        token_mlp_dim=256,
        channel_mlp_dim=512,
        num_layers=2,
        dropout=0.1,
    )


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        x_rot = x[..., : self.dim]
        x_rest = x[..., self.dim :] if x.shape[-1] > self.dim else None

        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        x_rotated = x_rot * cos + rotated * sin

        if x_rest is not None:
            return torch.cat([x_rotated, x_rest], dim=-1)
        return x_rotated


class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.proj_in = nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.spatial_proj = CausalLinear(seq_len, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)
        u, v = x.chunk(2, dim=1)
        v = self.norm(v.transpose(1, 2)).transpose(1, 2)
        v = self.spatial_proj(v)
        v = self.dropout(v)
        x = u * v
        x = x.transpose(1, 2)
        return x


class HyperMixing(nn.Module):
    def __init__(self, hidden_dim: int, hyper_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hyper = nn.Sequential(
            nn.Linear(hidden_dim, hyper_hidden_dim),
            nn.GELU(),
            nn.Linear(hyper_hidden_dim, hidden_dim * 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        cumsum = torch.cumsum(x, dim=1)
        counts = torch.arange(1, S + 1, device=x.device).view(1, S, 1).float()
        pooled = cumsum / counts

        weights = self.hyper(pooled)
        w1, w2 = weights.chunk(2, dim=-1)

        x = x * w1 + w2
        return self.dropout(x)


class ImprovedMixerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        channel_mlp_dim: int,
        use_hyper: bool = False,
        hyper_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        if use_hyper:
            hhd = hyper_hidden_dim if hyper_hidden_dim is not None else hidden_dim // 2
            self.token_mixer = HyperMixing(hidden_dim, hhd, dropout)
        else:
            self.token_mixer = SpatialGatingUnit(hidden_dim, seq_len, dropout)

        self.channel_mlp = MlpBlock(hidden_dim, channel_mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.token_mixer(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.channel_mlp(x)
        x = x + residual

        return x


@dataclass
class MicroMixerV2Config:
    vocab_size: int = 256
    max_seq_len: int = 128
    hidden_dim: int = 128
    channel_mlp_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    tie_weights: bool = True
    use_hyper: bool = False
    hyper_hidden_dim: int = 64


class MicroMixerV2(nn.Module):
    def __init__(self, config: MicroMixerV2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.hidden_dim = config.hidden_dim

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.rope = RotaryPositionEmbedding(config.hidden_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

        self.mixer_layers = nn.ModuleList([
            ImprovedMixerLayer(
                config.hidden_dim,
                config.max_seq_len,
                config.channel_mlp_dim,
                config.use_hyper,
                config.hyper_hidden_dim,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if getattr(config, "tie_weights", True):
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, CausalLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, S = input_ids.shape

        if S > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]
            S = self.max_seq_len
            if targets is not None:
                targets = targets[:, -self.max_seq_len :]

        token_emb = self.token_embedding(input_ids)
        x = self.rope(token_emb, S)
        x = self.dropout(x)

        for layer in self.mixer_layers:
            x = layer(x)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )
            return logits, loss

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        for _ in range(max_new_tokens):
            logits = self(input_ids)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def micromixer_v2_100k() -> MicroMixerV2Config:
    return MicroMixerV2Config(
        max_seq_len=64,
        hidden_dim=84,
        channel_mlp_dim=128,
        num_layers=3,
        dropout=0.1,
    )


def micromixer_v2_300k() -> MicroMixerV2Config:
    return MicroMixerV2Config(
        max_seq_len=128,
        hidden_dim=128,
        channel_mlp_dim=288,
        num_layers=3,
        dropout=0.1,
    )


def micromixer_v2_500k() -> MicroMixerV2Config:
    return MicroMixerV2Config(
        max_seq_len=128,
        hidden_dim=176,
        channel_mlp_dim=384,
        num_layers=3,
        dropout=0.1,
    )


def micromixer_v2_1m() -> MicroMixerV2Config:
    return MicroMixerV2Config(
        max_seq_len=256,
        hidden_dim=224,
        channel_mlp_dim=576,
        num_layers=3,
        dropout=0.1,
    )


if __name__ == "__main__":
    for name, config_fn in [
        ("100k", micromixer_100k),
        ("300k", micromixer_300k),
        ("500k", micromixer_500k),
        ("1M", micromixer_1m),
    ]:
        config = config_fn()
        model = MicroMixerV1(config)
        params = count_parameters(model)
        print(f"V1 {name}: {params:,} parameters")

        batch_size = 2
        seq_len = min(32, config.max_seq_len)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, loss = model(input_ids, targets)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss.dim() == 0

        prompt = input_ids[:, :4]
        gen_ids = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=10)
        assert gen_ids.shape == (batch_size, 12)

        prefix = torch.randint(0, config.vocab_size, (1, 5))
        extra = torch.randint(0, config.vocab_size, (1, 3))
        logits_prefix = model(prefix)[:, -1, :]
        logits_extended = model(torch.cat([prefix, extra], dim=1))[:, 4, :]
        assert torch.allclose(logits_prefix, logits_extended, atol=1e-5)

    print("V1 tests passed!")

    for name, config_fn in [
        ("100k", micromixer_v2_100k),
        ("300k", micromixer_v2_300k),
        ("500k", micromixer_v2_500k),
        ("1M", micromixer_v2_1m),
    ]:
        config = config_fn()
        model = MicroMixerV2(config)
        params = count_parameters(model)
        print(f"V2 SGU {name}: {params:,} parameters")

        batch_size = 2
        seq_len = min(32, config.max_seq_len)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, loss = model(input_ids, targets)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss.dim() == 0

        prompt = input_ids[:, :4]
        gen_ids = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=10)
        assert gen_ids.shape == (batch_size, 12)

        prefix = torch.randint(0, config.vocab_size, (1, 5))
        extra = torch.randint(0, config.vocab_size, (1, 3))
        logits_prefix = model(prefix)[:, -1, :]
        logits_extended = model(torch.cat([prefix, extra], dim=1))[:, 4, :]
        assert torch.allclose(logits_prefix, logits_extended, atol=1e-5)

    print("V2 SGU tests passed!")

    for name, config_fn in [
        ("100k", micromixer_v2_100k),
        ("300k", micromixer_v2_300k),
        ("500k", micromixer_v2_500k),
        ("1M", micromixer_v2_1m),
    ]:
        config = config_fn()
        config.use_hyper = True
        model = MicroMixerV2(config)
        params = count_parameters(model)
        print(f"V2 Hyper {name}: {params:,} parameters")

        batch_size = 2
        seq_len = min(32, config.max_seq_len)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, loss = model(input_ids, targets)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss.dim() == 0

        prompt = input_ids[:, :4]
        gen_ids = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=10)
        assert gen_ids.shape == (batch_size, 12)

        prefix = torch.randint(0, config.vocab_size, (1, 5))
        extra = torch.randint(0, config.vocab_size, (1, 3))
        logits_prefix = model(prefix)[:, -1, :]
        logits_extended = model(torch.cat([prefix, extra], dim=1))[:, 4, :]
        assert torch.allclose(logits_prefix, logits_extended, atol=1e-5)

    print("V2 HyperMixing tests passed!")

    config = micromixer_100k()
    model = MicroMixer(config)
    params = count_parameters(model)
    print(f"MicroMixer alias: {params:,} parameters")
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
    print("Alias test passed!")

    print("All tests passed!")
