#!/data/data/com.termux/files/usr/bin/bash
# =====================================================
# 🚀 ARSLM - Installation complète des dépendances
# =====================================================
# Option 3 : Installation complète
set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_header() {
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_header "ARSLM - Installation Complète"

# Vérifier qu'on est dans ARSLM
if [ ! -f "pyproject.toml" ]; then
    print_error "Pas dans le dossier ARSLM !"
    exit 1
fi

# ===============================
# PHASE 1 : Packages Termux
# ===============================
print_header "Phase 1/5 : Packages Système"

print_info "Mise à jour des repositories..."
pkg update -y 2>/dev/null || print_warning "Mise à jour partielle"

print_info "Installation des packages système..."
pkg install -y \
    python \
    python-pip \
    build-essential \
    clang \
    libffi \
    openssl \
    git \
    wget \
    2>/dev/null || print_warning "Installation partielle"

print_success "Packages système installés"

# ===============================
# PHASE 2 : pip et outils de base
# ===============================
print_header "Phase 2/5 : Outils de Base Python"

print_info "Mise à jour de pip..."
python -m pip install --upgrade pip
print_success "pip $(pip --version | cut -d' ' -f2)"

print_info "Installation setuptools, wheel..."
pip install --upgrade setuptools wheel
print_success "setuptools et wheel installés"

# ===============================
# PHASE 3 : Outils de build
# ===============================
print_header "Phase 3/5 : Outils de Build"

print_info "Installation de build..."
pip install build
print_success "build installé"

print_info "Installation de twine..."
pip install twine
print_success "twine installé"

# ===============================
# PHASE 4 : Dépendances requirements-dev.txt
# ===============================
print_header "Phase 4/5 : Dépendances de Développement"

if [ -f "requirements-dev.txt" ]; then
    print_info "Installation depuis requirements-dev.txt..."
    
    # Installer ligne par ligne pour mieux voir les erreurs
    while IFS= read -r line; do
        # Ignorer les commentaires et lignes vides
        if [[ ! "$line" =~ ^# ]] && [[ -n "$line" ]] && [[ ! "$line" =~ ^-r ]]; then
            package=$(echo "$line" | cut -d'>' -f1 | cut -d'=' -f1 | cut -d'<' -f1 | xargs)
            if [ -n "$package" ]; then
                print_info "Installation de $package..."
                pip install "$line" 2>/dev/null && print_success "$package" || print_warning "$package (échec, continué)"
            fi
        fi
    done < requirements-dev.txt
    
    print_success "Dépendances de dev installées"
else
    print_warning "requirements-dev.txt non trouvé, installation manuelle..."
    
    # Installation manuelle des packages essentiels
    print_info "Installation des packages essentiels..."
    
    # Testing
    pip install pytest pytest-cov pytest-asyncio pytest-mock || print_warning "Pytest partiellement installé"
    
    # Code quality
    pip install black isort flake8 || print_warning "Outils qualité partiellement installés"
    
    # Build tools
    pip install build twine wheel || print_warning "Build tools partiellement installés"
    
    print_success "Installation manuelle terminée"
fi

# ===============================
# PHASE 5 : Installation du package en mode dev
# ===============================
print_header "Phase 5/5 : Installation Package ARSLM"

print_info "Installation de ARSLM en mode développement..."
if pip install -e . 2>/dev/null; then
    print_success "ARSLM installé en mode dev"
else
    print_warning "Installation en mode dev échouée (peut-être normal)"
    print_info "Essai avec requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt || print_warning "Installation partielle"
    fi
fi

# ===============================
# VÉRIFICATIONS
# ===============================
print_header "Vérification des Installations"

echo ""
print_info "Outils de build :"
python -m build --version 2>/dev/null && print_success "  build OK" || print_error "  build MANQUANT"
twine --version 2>/dev/null && print_success "  twine OK" || print_error "  twine MANQUANT"

echo ""
print_info "Outils de qualité :"
black --version 2>/dev/null && print_success "  black OK" || print_warning "  black manquant"
isort --version 2>/dev/null && print_success "  isort OK" || print_warning "  isort manquant"
flake8 --version 2>/dev/null && print_success "  flake8 OK" || print_warning "  flake8 manquant"

echo ""
print_info "Outils de test :"
pytest --version 2>/dev/null && print_success "  pytest OK" || print_warning "  pytest manquant"

echo ""
print_info "Package ARSLM :"
python -c "import arslm; print(f'  Version: {arslm.__version__}')" 2>/dev/null && print_success "  arslm importable" || print_warning "  arslm non importable (peut-être normal)"

# ===============================
# TEST DE BUILD
# ===============================
print_header "Test de Construction du Package"

print_info "Nettoyage des anciennes builds..."
rm -rf build/ dist/ *.egg-info 2>/dev/null || true

print_info "Construction du package..."
if python -m build 2>&1 | tee build.log; then
    echo ""
    print_success "BUILD RÉUSSI !"
    echo ""
    print_info "Fichiers créés :"
    ls -lh dist/ 2>/dev/null || print_warning "Dossier dist/ vide"
    
    echo ""
    print_info "Vérification avec twine..."
    if twine check dist/* 2>/dev/null; then
        print_success "Package valide pour PyPI !"
    else
        print_warning "Vérification twine échouée"
    fi
else
    echo ""
    print_error "BUILD ÉCHOUÉ"
    print_info "Voir build.log pour plus de détails"
fi

# ===============================
# RÉSUMÉ FINAL
# ===============================
print_header "Résumé de l'Installation"

echo ""
print_success "Installation terminée !"
echo ""

# Compter les packages installés
installed_count=$(pip list | wc -l)
print_info "Packages Python installés : $installed_count"

echo ""
print_info "Commandes disponibles :"
echo "  python -m build          # Construire le package"
echo "  twine check dist/*       # Vérifier le package"
echo "  pytest tests/            # Lancer les tests"
echo "  black arslm/ tests/      # Formater le code"
echo "  flake8 arslm/ tests/     # Linter le code"
echo ""

print_info "Fichiers importants :"
echo "  build.log                # Log de construction"
echo "  dist/                    # Packages construits"
echo ""

# Vérifier si prêt pour PyPI
if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
    print_success "🎉 PRÊT POUR PYPI !"
    echo ""
    print_info "Prochaines étapes :"
    echo "  1. twine upload --repository testpypi dist/*"
    echo "  2. Tester l'installation depuis TestPyPI"
    echo "  3. twine upload dist/*  # Production"
else
    print_warning "Package non construit, réessayez : python -m build"
fi

echo ""
print_success "✨ Installation complète terminée !"
echo ""