<div align="center">

<img src="../../logo.svg" width="300" alt="MicroMixer-1 Logo"/>

# MicroMixer-1-1M-TinyStories

<img src="https://img.shields.io/badge/Parameters-967%2C584-blue?style=for-the-badge&logo=python&logoColor=white&color=%23007BFF" alt="Parameters"/>
<img src="https://img.shields.io/badge/Architecture-MLP--Mixer-purple?style=for-the-badge&color=%23AE00FF" alt="Architecture"/>
<img src="https://img.shields.io/badge/Dataset-TinyStories-green?style=for-the-badge&color=%2300D620" alt="Dataset"/>

<br/>
<br/>

<table>
<tr>
<td align="center" style="padding: 20px;">
<strong style="color: #007BFF; font-size: 1.2em;">Micro Language Model</strong><br/>
<em>Attention-Free • MLP-Only • Byte-Level</em>
</td>
</tr>
</table>

[![GitHub](https://img.shields.io/badge/GitHub-MicroMixer--1-blue?style=for-the-badge&logo=github&color=%23007BFF)](https://github.com/llaa33219/MicroMixer-1)

</div>

---

<div style="background: linear-gradient(135deg, #007BFF22, #AE00FF22); padding: 20px; border-radius: 10px; border-left: 4px solid #007BFF;">

## 📋 Overview

**MicroMixer-1-1M** is the largest model in the series with ~1M parameters. It can generate sentences like "Once upon a time there was a little girl named Lily" with reasonable fluency. Supports the longest sequence length of 256 tokens.

</div>

---

## 🏗️ Architecture

<div align="center">

```mermaid
graph TD
    A[Byte Input] --> B[Token Embedding]
    B --> C[RoPE Position Encoding]
    C --> D[ImprovedMixerLayer ×3]
    D --> E[LayerNorm]
    E --> F[LM Head]
    F --> G[Byte Output]
    
    style A fill:#007BFF,color:#fff
    style G fill:#00D620,color:#fff
    style D fill:#AE00FF,color:#fff
```

</div>

### Model Configuration

<table>
<tr>
<th style="background-color: #007BFF; color: white;">Parameter</th>
<th style="background-color: #AE00FF; color: white;">Value</th>
</tr>
<tr><td>Total Parameters</td><td><code>967,584</code></td></tr>
<tr><td>Hidden Dimension</td><td><code>224</code></td></tr>
<tr><td>Channel MLP Dimension</td><td><code>576</code></td></tr>
<tr><td>Number of Layers</td><td><code>3</code></td></tr>
<tr><td>Max Sequence Length</td><td><code>256</code></td></tr>
<tr><td>Vocabulary Size</td><td><code>256</code> (Byte-level)</td></tr>
</table>

### Core Components

<div style="background-color: #1a1a2e; padding: 15px; border-radius: 8px;">

```
┌─────────────────────────────────────────────┐
│           ImprovedMixerLayer                 │
│  ┌─────────────────────────────────────┐    │
│  │  LayerNorm → HyperMixing → Residual │    │ ← Token Mixing
│  ├─────────────────────────────────────┤    │
│  │  LayerNorm → MlpBlock → Residual    │    │ ← Channel Mixing
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

</div>

#### 1️⃣ RoPE (Rotary Position Embedding)
- Encodes positions via **rotation transformations**
- Enables length extrapolation beyond training sequences

#### 2️⃣ HyperMixing (Token Mixing)
- Compresses past context via **cumulative average pooling**
- Hypernetwork generates adaptive weights
- O(S) complexity token mixing without attention

#### 3️⃣ MlpBlock (Channel Mixing)
- Non-linear transformation of feature dimensions
- Structure: `Linear → GELU → Linear`

---

## 📈 Key Differences from 500K

<div style="background-color: #FFEA0015; padding: 15px; border-radius: 8px; border-left: 4px solid #FFEA00;">

| Metric | 500K | 1M | Change |
|--------|------|-----|--------|
| Parameters | 557,328 | 967,584 | **1.7x** |
| Hidden Dim | 176 | 224 | 1.3x |
| Channel MLP | 384 | 576 | 1.5x |
| Sequence Length | 128 | 256 | **2x** |

</div>

---

## 🎯 Generation Examples

<div style="background-color: #AE00FF15; padding: 15px; border-radius: 8px; border-left: 4px solid #AE00FF;">

```
Prompt: "Once upon a time"
Output: "Once upon a time there was a little girl named Lily. She loved t..."

Prompt: "Hello"
Output: "Hellog ann he grit litle girls. She love loved in the"
```

</div>

---

## 📊 Model Comparison

<div style="background-color: #007BFF15; padding: 15px; border-radius: 8px; border-left: 4px solid #007BFF;">

| Model | Parameters | Hidden | Seq Len | Quality |
|-------|------------|--------|---------|---------|
| 100K | 136,908 | 84 | 64 | ⭐ |
| 300K | 331,680 | 128 | 128 | ⭐⭐ |
| 500K | 557,328 | 176 | 128 | ⭐⭐⭐ |
| **1M** | **967,584** | **224** | **256** | ⭐⭐⭐⭐ |

</div>

---

## ⚠️ Limitations

<div style="background-color: #FF050515; padding: 15px; border-radius: 8px; border-left: 4px solid #FF0505;">

| Limitation | Description |
|------------|-------------|
| **Grammatical Errors** | Fully grammatical sentences still difficult |
| **Name Instability** | Same name varies: "Lily", "Limmy", "Amby" |
| **Short Prompt Issues** | "Hello", "The weather is" produce near-random output |
| **Overfitting** | Overfits to specific TinyStories phrases |

</div>

---

## 📊 Training Data

<div style="background-color: #00D62015; padding: 15px; border-radius: 8px; border-left: 4px solid #00D620;">

**Dataset**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

- Simple children's stories dataset
- Learns basic grammar and vocabulary
- Contains many patterns like "Once upon a time", "little girl/boy"

</div>

---

## 🔧 Usage

```python
import torch
from src.model import MicroMixerV2, MicroMixerV2Config
from src.tokenizer import ByteTokenizer

# Clone the repository first:
# git clone https://github.com/llaa33219/MicroMixer-1.git
# cd MicroMixer-1

config = MicroMixerV2Config(
    max_seq_len=256,
    hidden_dim=224,
    channel_mlp_dim=576,
    num_layers=3,
    use_hyper=True,
)

model = MicroMixerV2(config)
checkpoint = torch.load("checkpoints/v2_hyper_1M/epoch_4.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = ByteTokenizer()
input_ids = torch.tensor([tokenizer.encode("Once upon a time")])

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=128, temperature=0.8, top_k=40)

print(tokenizer.decode(output[0].tolist()))
```

---

<div align="center">

[![GitHub](https://img.shields.io/badge/Back_to_Repository-MicroMixer--1-blue?style=for-the-badge&logo=github&color=%23007BFF)](https://github.com/llaa33219/MicroMixer-1)

<sub>Part of the <a href="https://github.com/llaa33219/MicroMixer-1">MicroMixer-1</a> research project</sub>

</div>
