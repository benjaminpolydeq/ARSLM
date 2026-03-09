# 🧠 ARSLM — Adaptive Recurrent State Language Model

![Version](https://img.shields.io/badge/version-1.0.0--MVP-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-MVP-yellow)

**A sequential language model combining adaptive recurrent networks and attention mechanisms —  
designed for on-premise deployment in businesses and institutions worldwide.**

---

## 🌟 Overview

**ARSLM** (**A**daptive **R**ecurrent **S**tate **L**anguage **M**odel) is a lightweight, modular AI engine that adapts in real time to dynamic data streams. It combines:

- 🔁 **Adaptive Recurrent Networks** — captures long-range dependencies across sequences
- 🎯 **Attention Mechanisms** — focuses on the most relevant context at each step
- 🧠 **RAG (Retrieval-Augmented Generation)** — grounds responses in real documents via FAISS
- 🔧 **LoRA Fine-tuning** — domain adaptation with minimal compute

ARSLM runs entirely **on-premise** — your data never leaves your infrastructure.

---

## 🗂️ Project Structure

```
ARSLM/
├── arslm/
│   ├── __init__.py        ← Package init (exports ARSLMEngine, RAGIndex, ARSLMTrainer)
│   ├── engine.py          ← Core AI engine: model loading, LoRA, RAG, generation
│   └── trainer.py         ← LoRA fine-tuning pipeline (SFTTrainer)
├── microllm_core.py       ← Session & conversation manager
├── app.py                 ← Streamlit web interface
├── main.py                ← CLI entry point (launch / train / index)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/benjaminpolydeq/ARSLM.git
cd ARSLM

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Quick Start

### Launch the web interface

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Launch via CLI (with optional training)

```bash
# Just launch the UI
python main.py

# Fine-tune LoRA then launch
python main.py --train

# Build RAG index only
python main.py --index-only
```

---

## ⚙️ Configuration

All settings are controlled via environment variables (create a `.env` file):

```env
ARSLM_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
ARSLM_MODEL_PATH=./arslm_lora
ARSLM_INDEX_PATH=./arslm_index
ARSLM_DRIVE_PATH=/content/drive/MyDrive/ARSLM/arslm_lora
```

---

## 🐍 Python API

```python
from arslm import ARSLMEngine

engine = ARSLMEngine(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    index_path="./arslm_index",
)

response = engine.generate(
    prompt="What are the rights of an accused person?",
    domain="legal",
    max_new_tokens=300,
)
print(response)
```

### With session management

```python
from microllm_core import MicroLLMCore

core = MicroLLMCore(engine)
reply = core.chat("Explain custody procedure", session_id="user42", domain="police")
print(reply)
print(core.get_history("user42"))
```

### Fine-tuning

```python
from arslm import ARSLMTrainer
from datasets import Dataset

trainer = ARSLMTrainer(engine=engine, output_dir="./arslm_lora")

# Load wikitext (no /tmp/ dependency — safe across Colab sessions)
passages = ARSLMTrainer.load_wikitext(n=10_000)
engine.rag.build_index(passages)

# Fine-tune on domain data
data = Dataset.from_list([
    {"text": "[Domain: legal] Q: ... A: ..."},
])
trainer.train(data, epochs=3)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           Streamlit Interface  (app.py)          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│        MicroLLM Core  (microllm_core.py)         │
│    Session Manager  │  Conversation History      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│          ARSLM Engine  (arslm/engine.py)         │
│                                                  │
│  ┌─────────────┐  ┌───────────┐  ┌───────────┐  │
│  │  Recurrent  │  │ Attention │  │    RAG    │  │
│  │    State    │→ │ Mechanism │→ │  (FAISS)  │  │
│  │   (LoRA)    │  │           │  │           │  │
│  └─────────────┘  └───────────┘  └───────────┘  │
└─────────────────────────────────────────────────┘
```

| Component | Role |
|---|---|
| **Recurrent State (LoRA)** | Adapts the base LLM to your domain via lightweight fine-tuning |
| **Attention Mechanism** | Focuses generation on the most relevant tokens in context |
| **RAG / FAISS** | Retrieves real documents to ground and enrich every response |
| **MicroLLM Core** | Manages user sessions and conversation history |
| **Streamlit UI** | Simple, secure web interface — runs locally |

---

## 🔒 Security Notes

- **No public tunnel**: the Streamlit interface runs locally (`localhost:8501`)
- **Secrets via env vars**: never hardcode credentials in source code
- **Model weights excluded from Git**: add `arslm_lora/` to `.gitignore` ✅

---

## 📄 License

MIT License — Copyright © 2025 Benjamin Amaad Kama.  
See [LICENSE](LICENSE) for full terms.

---

**Made with ❤️ by Benjamin Amaad Kama**
