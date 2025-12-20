# ARSLM Architecture Documentation

## 🏗️ System Overview

ARSLM (Adaptive Reasoning Semantic Language Model) is built on a modular, scalable architecture designed for production deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│                                                              │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   Streamlit   │              │  REST API    │            │
│  │   Frontend    │              │   (FastAPI)  │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼──────────────────┘
          │                              │
┌─────────▼──────────────────────────────▼──────────────────┐
│                    Application Layer                        │
│                                                             │
│  ┌────────────────┬──────────────────┬──────────────────┐ │
│  │  Session       │  Conversation    │  Response        │ │
│  │  Management    │  Handler         │  Generator       │ │
│  └────────────────┴──────────────────┴──────────────────┘ │
└─────────────────────────┬─────────────────────────────────┘
                          │
┌─────────────────────────▼─────────────────────────────────┐
│                      AI Core Layer                         │
│                                                            │
│  ┌──────────────┬──────────────┬──────────────────────┐  │
│  │  ARSLM Model │  Tokenizer   │  Inference Engine    │  │
│  │  (PyTorch)   │              │                      │  │
│  └──────────────┴──────────────┴──────────────────────┘  │
└─────────────────────────┬─────────────────────────────────┘
                          │
┌─────────────────────────▼─────────────────────────────────┐
│                      Data Layer                            │
│                                                            │
│  ┌──────────────┬──────────────┬──────────────────────┐  │
│  │  PostgreSQL  │    Redis     │  File Storage        │  │
│  │  (metadata)  │   (cache)    │  (models/data)       │  │
│  └──────────────┴──────────────┴──────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Components

### 1. ARSLM Model

**Location**: `src/arslm/core/model.py`

The core neural architecture based on transformers with adaptive mechanisms.

#### Key Features

- **Adaptive Attention**: Dynamic attention weights based on context
- **Efficient Architecture**: Optimized for inference speed
- **Modular Design**: Easy to extend and customize

#### Model Architecture

```python
ARSLMModel(
  (token_embedding): Embedding(50000, 512)
  (position_embedding): Embedding(512, 512)
  (layers): ModuleList(
    (0-5): 6 x EncoderLayer(
      (self_attn): AdaptiveAttention(
        (W_q): Linear(512, 512)
        (W_k): Linear(512, 512)
        (W_v): Linear(512, 512)
        (W_o): Linear(512, 512)
        (gate): Linear(512, 8)
      )
      (feed_forward): FeedForward(
        (linear1): Linear(512, 2048)
        (linear2): Linear(2048, 512)
      )
      (norm1): LayerNorm(512)
      (norm2): LayerNorm(512)
    )
  )
  (output_projection): Linear(512, 50000)
)
```

#### Configuration

```python
@dataclass
class ARSLMConfig:
    vocab_size: int = 50000        # Vocabulary size
    d_model: int = 512             # Model dimension
    n_heads: int = 8               # Number of attention heads
    n_layers: int = 6              # Number of encoder layers
    d_ff: int = 2048               # Feed-forward dimension
    dropout: float = 0.1           # Dropout rate
    max_seq_length: int = 512      # Maximum sequence length
```

---

### 2. Adaptive Attention Mechanism

**Location**: `src/arslm/core/model.py`

Novel attention mechanism with dynamic gating.

#### How It Works

1. **Standard Multi-Head Attention**
   ```python
   Q = W_q(x)  # Query projection
   K = W_k(x)  # Key projection
   V = W_v(x)  # Value projection
   ```

2. **Adaptive Gating**
   ```python
   gate = sigmoid(W_gate(x))  # Per-head gates
   attention = attention * gate  # Modulate attention
   ```

3. **Benefits**
   - Dynamic focus on relevant information
   - Improved long-range dependencies
   - Better context understanding

---

### 3. API Server

**Location**: `src/api/main.py`

FastAPI-based REST API for model inference.

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/chat` | POST | Generate response |
| `/api/v1/history/{session_id}` | GET | Get conversation history |
| `/api/v1/history/{session_id}` | DELETE | Clear history |
| `/api/v1/sessions` | GET | List active sessions |
| `/api/v1/model/info` | GET | Model information |

#### Request/Response Flow

```
Client Request
     ↓
FastAPI Middleware (CORS, Auth)
     ↓
Request Validation (Pydantic)
     ↓
Session Management
     ↓
Tokenization
     ↓
Model Inference
     ↓
Response Generation
     ↓
JSON Response
```

---

### 4. Frontend Interface

**Location**: `src/frontend/app.py`

Streamlit-based chat interface.

#### Components

- **Chat Interface**: Main conversation area
- **Sidebar**: Settings and controls
- **History View**: Past conversations
- **Parameter Controls**: Generation settings

#### State Management

```python
# Session state
- session_id: str           # Unique session identifier
- messages: List[Dict]      # Conversation history
- api_available: bool       # API connection status
```

---

## 📊 Data Flow

### Inference Pipeline

```
User Input
    ↓
Frontend (Streamlit)
    ↓ HTTP POST
API Server (FastAPI)
    ↓
Tokenizer
    ↓ token_ids
ARSLM Model
    ↓ logits
Sampling (top-k, top-p)
    ↓ token_ids
Tokenizer (decode)
    ↓ text
Response
```

### Training Pipeline (Future)

```
Training Data
    ↓
Data Loader
    ↓ batches
Model Forward Pass
    ↓ logits
Loss Calculation
    ↓
Backpropagation
    ↓
Optimizer Step
    ↓
Checkpoint Save
```

---

## 🗄️ Data Storage

### Conversation Storage

**Current**: In-memory dictionary
**Future**: PostgreSQL with schema:

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_session ON conversations(session_id);
CREATE INDEX idx_timestamp ON conversations(timestamp);
```

### Model Storage

**Location**: `models/`

```
models/
├── arslm_base.pt           # Base model checkpoint
├── arslm_large.pt          # Large model
└── checkpoints/
    ├── checkpoint_001.pt
    └── checkpoint_002.pt
```

**Format**: PyTorch checkpoint
```python
{
    'model_state_dict': OrderedDict(...),
    'config': ARSLMConfig(...),
    'optimizer_state_dict': OrderedDict(...),  # Optional
    'epoch': int,
    'loss': float
}
```

---

## 🔧 Configuration Management

### Environment Variables

**File**: `.env`

```bash
# Application
ENVIRONMENT=development|production
DEBUG=true|false

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model
MODEL_PATH=./models/arslm_base.pt
MODEL_MAX_LENGTH=512

# Database
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Python Configuration

**File**: `config/settings.py`

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "ARSLM"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
```

---

## 🚀 Deployment Architecture

### Docker Compose Stack

```yaml
services:
  - api         # FastAPI application
  - frontend    # Streamlit interface  
  - postgres    # Database
  - redis       # Cache
  - nginx       # Reverse proxy
```

### Production Deployment

```
Internet
    ↓
Load Balancer
    ↓
    ├── API Server 1
    ├── API Server 2
    └── API Server 3
         ↓
    Shared Database
```

---

## 🔐 Security Architecture

### Authentication Flow

```
User Request
    ↓
API Key/JWT Token
    ↓
Verification
    ↓ (if valid)
Request Processing
```

### Data Security

- **Encryption at Rest**: AES-256
- **Encryption in Transit**: TLS 1.3
- **Input Validation**: Pydantic schemas
- **Rate Limiting**: Token bucket algorithm

---

## 📈 Scalability

### Horizontal Scaling

- **Stateless API**: Multiple API instances
- **Load Balancing**: Nginx/HAProxy
- **Shared Storage**: PostgreSQL + Redis

### Vertical Scaling

- **GPU Support**: CUDA-enabled inference
- **Batch Processing**: Multiple requests together
- **Model Quantization**: Reduce memory usage

---

## 🔍 Monitoring & Observability

### Metrics

- Request latency
- Throughput (requests/sec)
- Model inference time
- Memory usage
- Error rates

### Logging

```python
# Structured logging
{
    "timestamp": "2025-12-20T10:30:00Z",
    "level": "INFO",
    "service": "api",
    "event": "inference_request",
    "session_id": "abc123",
    "duration_ms": 42.5
}
```

---

## 🧪 Testing Architecture

### Test Pyramid

```
        /\
       /  \
      /E2E \
     /------\
    /  Intg  \
   /----------\
  /    Unit    \
 /--------------\
```

- **Unit Tests**: Individual components
- **Integration Tests**: API endpoints
- **E2E Tests**: Complete workflows

---

## 🔄 Development Workflow

```
Feature Branch
    ↓
Local Development
    ↓
Unit Tests (pytest)
    ↓
Code Review (PR)
    ↓
CI/CD (GitHub Actions)
    ↓
    ├── Lint (black, flake8)
    ├── Tests (pytest)
    └── Build (Docker)
    ↓
Staging Deployment
    ↓
Production Deployment
```

---

## 📚 Technology Stack

### Core
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Transformers**: 4.30+

### API
- **FastAPI**: 0.100+
- **Uvicorn**: 0.23+
- **Pydantic**: 2.0+

### Frontend
- **Streamlit**: 1.25+

### Data
- **PostgreSQL**: 15+
- **Redis**: 7+

### DevOps
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **GitHub Actions**: CI/CD

---

## 🎯 Future Improvements

### Short Term
- [ ] Add authentication
- [ ] Implement proper database
- [ ] Add metrics/monitoring
- [ ] Improve error handling

### Medium Term
- [ ] Multi-language support
- [ ] Fine-tuning API
- [ ] Model versioning
- [ ] A/B testing framework

### Long Term
- [ ] Distributed training
- [ ] Multi-modal support
- [ ] Auto-scaling
- [ ] Edge deployment

---

## 📞 Contact

For architecture questions:
- Email: benjokama@hotmail.fr
- GitHub: [@benjaminpolydeq](https://github.com/benjaminpolydeq)
