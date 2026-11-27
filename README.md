# 🧠 ARSLM — Adaptive Recurrent State Language Model

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-experimental-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

**A lightweight language model built from scratch**, combining adaptive recurrent networks and attention mechanisms.

> This is **not** a wrapper around Mistral, Llama or any external LLM.  
> ARSLM is a custom hybrid architecture designed for research, education and lightweight on-device / on-premise experiments.

---

## Why from-scratch?

Most small projects today just fine-tune or wrap existing large models.  
ARSLM takes the opposite approach:

- Full control over the architecture
- Transparent and educational codebase
- Suitable for experimentation with hybrid Recurrent + Attention designs
- Small enough to train and run on modest hardware

---

## Architecture

```
Input tokens
    ↓
Token + Position Embeddings
    ↓
┌─────────────────────────────────────┐
│  ARSLM Block  (× N layers)          │
│  ┌─────────────┐   ┌──────────────┐ │
│  │  Adaptive   │ → │  Attention   │ │
│  │  Recurrent  │   │  (Multi-Head │ │
│  │  (LSTM/GRU) │   │   / Adaptive)│ │
│  └─────────────┘   └──────────────┘ │
│           ↓ residual + FFN          │
└─────────────────────────────────────┘
    ↓
Final LayerNorm → LM Head → logits
```

Key ideas:
- **Adaptive Recurrent blocks** capture sequential state
- **Attention** (standard, adaptive or local) provides content-based focus
- Residual connections + LayerNorm for stable training

---

## Project structure (clean target)

```
arslm/
├── __init__.py          # exports ARSLM, ARSLMConfig
├── config.py            # hyperparameters
├── model.py             # main from-scratch model
└── core/
    ├── attention.py     # MultiHead, Adaptive, Local attention
    └── recurrent.py     # Adaptive LSTM / GRU / RNN
```

---

## Quick start

```bash
git clone https://github.com/benjaminpolydeq/ARSLM.git
cd ARSLM

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch
# (optional) pip install -e .
```

### Create a small model and run a forward pass

```python
import torch
from arslm import ARSLM, ARSLMConfig

config = ARSLMConfig(
    vocab_size=1000,
    hidden_size=128,
    num_layers=2,
    num_attention_heads=4,
    max_position_embeddings=128,
)

model = ARSLM(config)
print(f"Parameters: {model.get_num_params():,}")

input_ids = torch.randint(0, 1000, (2, 32))  # batch=2, seq=32
outputs = model(input_ids)
print(outputs["logits"].shape)  # (2, 32, 1000)
```

### Simple generation

```python
prompt = torch.randint(0, 1000, (1, 10))
generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
print(generated.shape)
```

---

## Current status

| Component              | Status          |
|------------------------|-----------------|
| Core attention modules | ✅ Implemented  |
| Core recurrent modules | ✅ Implemented  |
| Main `ARSLM` model     | ✅ From-scratch |
| Config system          | ✅ Done         |
| Training loop          | 🚧 Planned      |
| Tokenizer              | 🚧 Planned      |
| Proper packaging       | 🚧 In progress  |
| Tests                  | 🚧 Partial      |

This repository is under active cleanup and refactoring toward a clean from-scratch design.

---

## Roadmap (short term)

1. Finish repository cleanup (remove legacy wrapper code & committed `.venv`)
2. Add a minimal training script
3. Add unit tests for the core blocks
4. Publish a small pre-trained toy checkpoint (optional)
5. Clean packaging (`pyproject.toml` + `pip install -e .`)

---

## License

MIT License — Copyright © 2025 Benjamin Amaad Kama.

---

**Built as a from-scratch experiment by Benjamin Amaad Kama**
