# Changelog

All notable changes to ARSLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-language support (Spanish, French, Arabic, Portuguese)
- Streamlit web interface
- Fine-tuning scripts and documentation
- Model quantization support
- ONNX export functionality
- Pre-trained model zoo

---

## [0.1.0] - 2025-01-XX (Initial Release)

### Added
- ✨ Core ARSLM model implementation with configurable architecture
- 🧠 Multi-head attention mechanism
- 🔄 Adaptive recurrent networks (LSTM, GRU, RNN)
- 🎯 Adaptive components (MoE, dynamic routing, ACT)
- 📝 ARSLMTokenizer for text processing
- 🌐 REST API with FastAPI
- 🖥️ Command-line interface (CLI)
- 📦 Python client for API interaction
- 🧪 Comprehensive test suite with pytest
- 📚 Complete documentation
- 🐳 Docker support
- 🔧 Configuration management
- 📊 Model save/load functionality

### Core Features
- Text generation with temperature, top-k, and top-p sampling
- Conversation context management
- Session-based chat history
- Configurable model architecture
- Modular component design

### API Endpoints
- `POST /api/v1/chat` - Chat completion
- `POST /api/v1/generate` - Text generation
- `GET /api/v1/history/{session_id}` - Get conversation history
- `DELETE /api/v1/history/{session_id}` - Clear history
- `GET /health` - Health check
- `GET /api/v1/model/info` - Model information

### CLI Commands
- `arslm generate` - Generate text from prompt
- `arslm serve` - Start API server
- `arslm chat` - Interactive chat
- `arslm info` - Display model information
- `arslm train-tokenizer` - Train custom tokenizer

### Documentation
- Comprehensive README with examples
- API reference documentation
- Installation guide
- Quick start guide
- Architecture overview
- Contributing guidelines

### Development Tools
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking
- pytest for testing
- pre-commit hooks

### Package
- Published on PyPI as `arslm`
- Available on TestPyPI for testing
- Supports Python 3.8+
- Minimal dependencies for core functionality

---

## Release Notes

### v0.1.0 - Initial MVP Release

This is the first public release of ARSLM (Adaptive Reasoning Semantic Language Model).

**Highlights:**
- Complete working implementation of the ARSLM architecture
- Production-ready REST API
- Easy-to-use Python client
- Comprehensive CLI tools
- Full test coverage
- Professional documentation

**Target Users:**
- Developers building conversational AI applications
- Businesses requiring on-premises AI solutions
- Researchers exploring efficient language models
- Startups in emerging markets

**Known Limitations:**
- Currently supports English only (multi-language in roadmap)
- Model requires training/fine-tuning for specific domains
- No pre-trained checkpoints included yet

**Future Development:**
The next releases will focus on:
1. Pre-trained model checkpoints
2. Multi-language support
3. Fine-tuning utilities
4. Performance optimizations
5. Extended documentation and tutorials

---

## Version History

### 0.1.0 (Initial Release)
- First public release
- Core functionality implemented
- Published to PyPI

---

## Upgrade Guide

### From Development to 0.1.0

If you've been using ARSLM from the GitHub repository:

```bash
# Uninstall old version
pip uninstall arslm

# Install from PyPI
pip install arslm
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to ARSLM.

---

## Support

- 📖 [Documentation](https://arslm.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/benjaminpolydeq/ARSLM/issues)
- 💬 [Discussions](https://github.com/benjaminpolydeq/ARSLM/discussions)
- 📧 Email: benjokama@hotmail.fr

---

**Note:** Dates in YYYY-MM-DD format. Versions follow [Semantic Versioning](https://semver.org/).