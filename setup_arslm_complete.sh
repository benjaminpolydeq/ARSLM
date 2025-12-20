#!/bin/bash
################################################################################
# ARSLM Complete Setup Script
# 
# Ce script crée TOUTE la structure du dépôt ARSLM
# Utilisation: bash setup_arslm_complete.sh
################################################################################

set -e  # Exit on error

echo "🚀 ARSLM Complete Setup"
echo "======================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Erreur: Ce script doit être exécuté à la racine d'un repo Git${NC}"
    echo "Initialiser un repo d'abord avec: git init"
    exit 1
fi

echo -e "${BLUE}📍 Répertoire actuel: $(pwd)${NC}"
echo ""

# Confirmation
echo -e "${YELLOW}⚠️  Ce script va créer/modifier des fichiers dans ce répertoire.${NC}"
read -p "Continuer? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Annulé."
    exit 1
fi

################################################################################
# 1. BACKUP
################################################################################
echo -e "${BLUE}📦 Création d'une branche de backup...${NC}"
BACKUP_BRANCH="backup-$(date +%Y%m%d-%H%M%S)"
git checkout -b "$BACKUP_BRANCH" 2>/dev/null || echo "Branche déjà existante"
git add . 2>/dev/null || true
git commit -m "backup before restructure" 2>/dev/null || echo "Rien à commiter"
git checkout main 2>/dev/null || git checkout master 2>/dev/null || echo "Pas de branche main/master"
echo -e "${GREEN}✅ Backup créé sur branche: $BACKUP_BRANCH${NC}"
echo ""

################################################################################
# 2. STRUCTURE DE DOSSIERS
################################################################################
echo -e "${BLUE}📁 Création de la structure de dossiers...${NC}"

# Dossiers principaux
mkdir -p src/arslm/{core,training,inference,utils}
mkdir -p src/api/{routes,schemas,middleware}
mkdir -p src/frontend/{pages,components}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p docs
mkdir -p config
mkdir -p scripts
mkdir -p data/sample
mkdir -p models/checkpoints
mkdir -p notebooks
mkdir -p docker
mkdir -p .github/{workflows,ISSUE_TEMPLATE}
mkdir -p logs

# __init__.py pour tous les packages Python
find src tests -type d -exec touch {}/__init__.py \; 2>/dev/null || true

echo -e "${GREEN}✅ Structure créée${NC}"
echo ""

################################################################################
# 3. FICHIERS DE CONFIGURATION
################################################################################
echo -e "${BLUE}⚙️  Création des fichiers de configuration...${NC}"

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints

# Environment
.env
.env.local

# Logs
*.log
logs/

# Database
*.db
*.sqlite3

# Models & Data
models/*.pth
models/*.pt
models/*.h5
data/raw/
data/processed/
!data/.gitkeep
!models/.gitkeep

# Tests
htmlcov/
.coverage
.pytest_cache/

# OS
Thumbs.db
EOF

# .env.example
cat > .env.example << 'EOF'
# Application
APP_NAME=ARSLM
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Frontend
FRONTEND_PORT=8501

# Model
MODEL_PATH=./models/arslm_base.pt
MODEL_MAX_LENGTH=512

# Database
DATABASE_URL=sqlite:///./arslm.db

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=change_this_in_production
JWT_SECRET_KEY=change_this_too
EOF

# requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.30.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
streamlit>=1.25.0
pydantic>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0
EOF

# requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0
ipython>=8.14.0
jupyter>=1.0.0
EOF

echo -e "${GREEN}✅ Fichiers de configuration créés${NC}"
echo ""

################################################################################
# 4. DOCUMENTATION
################################################################################
echo -e "${BLUE}📚 Création de la documentation...${NC}"

# CONTRIBUTING.md
cat > CONTRIBUTING.md << 'EOF'
# Contributing to ARSLM

Thank you for your interest in contributing! 🎉

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit: `git commit -m "feat: add feature"`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests

## Questions?

Contact: benjokama@hotmail.fr
EOF

# CHANGELOG.md
cat > CHANGELOG.md << 'EOF'
# Changelog

## [Unreleased]

### Added
- Complete repository structure
- Core ARSLM model
- FastAPI REST API
- Streamlit frontend
- Docker support
- CI/CD pipeline

## [1.0.0] - 2025-12-20

### Added
- Initial release
EOF

echo -e "${GREEN}✅ Documentation créée${NC}"
echo ""

################################################################################
# 5. DOCKER
################################################################################
echo -e "${BLUE}🐳 Création des fichiers Docker...${NC}"

# Dockerfile
cat > docker/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# docker-compose.yml
cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - ../.env
    volumes:
      - ../models:/app/models
      - ../logs:/app/logs
    restart: unless-stopped

  frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    command: streamlit run src/frontend/app.py --server.port=8501
    depends_on:
      - api
    restart: unless-stopped
EOF

echo -e "${GREEN}✅ Fichiers Docker créés${NC}"
echo ""

################################################################################
# 6. CI/CD
################################################################################
echo -e "${BLUE}🔄 Création des workflows CI/CD...${NC}"

cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest tests/ -v --cov=src

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install black isort flake8
      
      - name: Run linters
        run: |
          black --check src/ tests/
          isort --check-only src/ tests/
          flake8 src/ tests/
EOF

echo -e "${GREEN}✅ CI/CD configuré${NC}"
echo ""

################################################################################
# 7. FICHIERS PLACEHOLDER
################################################################################
echo -e "${BLUE}📝 Création des fichiers placeholder...${NC}"

# Créer des fichiers README dans les dossiers importants
echo "# Models Directory" > models/README.md
echo "# Data Directory" > data/README.md
echo "# Logs Directory" > logs/README.md

# .gitkeep pour dossiers vides
touch data/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep

echo -e "${GREEN}✅ Fichiers placeholder créés${NC}"
echo ""

################################################################################
# 8. INSTRUCTIONS
################################################################################
cat > NEXT_STEPS.md << 'EOF'
# 🎯 Prochaines Étapes

## ✅ Ce qui a été fait

- ✅ Structure de dossiers créée
- ✅ Fichiers de configuration générés
- ✅ Documentation de base créée
- ✅ Docker configuré
- ✅ CI/CD configuré

## 📋 À faire maintenant

1. **Copier le code source**
   
   Vous devez copier manuellement les fichiers suivants depuis les artifacts Claude:
   
   - `src/arslm/core/model.py`
   - `src/arslm/utils/tokenizer.py`
   - `src/api/main.py`
   - `src/frontend/app.py`
   - `tests/unit/test_model.py`
   - `pyproject.toml`
   - `setup.py`

2. **Créer l'environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Copier .env**
   ```bash
   cp .env.example .env
   # Éditer .env avec vos valeurs
   ```

5. **Tester**
   ```bash
   pytest
   ```

6. **Lancer l'API**
   ```bash
   uvicorn src.api.main:app --reload
   ```

7. **Lancer le frontend**
   ```bash
   streamlit run src/frontend/app.py
   ```

8. **Commiter et pousser**
   ```bash
   git add .
   git commit -m "feat: complete repository structure"
   git push origin main
   ```

## 📞 Support

Problème ? Contactez: benjokama@hotmail.fr
EOF

################################################################################
# 9. FIN
################################################################################
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✅ Setup terminé avec succès!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${YELLOW}📋 Prochaines étapes:${NC}"
echo ""
echo "1. Lire: cat NEXT_STEPS.md"
echo "2. Copier les fichiers de code depuis les artifacts Claude"
echo "3. Créer l'environnement: python -m venv venv"
echo "4. Activer: source venv/bin/activate"
echo "5. Installer: pip install -e ."
echo "6. Tester: pytest"
echo "7. Commiter: git add . && git commit -m 'feat: complete structure'"
echo "8. Pousser: git push origin main"
echo ""
echo -e "${BLUE}Backup disponible sur branche: ${BACKUP_BRANCH}${NC}"
echo ""
echo -e "${GREEN}Bon développement! 🚀${NC}"