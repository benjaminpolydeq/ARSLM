# 📦 Guide Complet de Publication ARSLM sur PyPI

## 🎯 Checklist Pré-Publication

### ✅ Phase 1 : Préparation du Code

- [ ] Tous les fichiers sont créés et en place
- [ ] Tests unitaires écrits et passent (100% de couverture souhaité)
- [ ] Code formaté avec black : `black arslm/ tests/`
- [ ] Imports triés avec isort : `isort arslm/ tests/`
- [ ] Pas d'erreurs flake8 : `flake8 arslm/ tests/`
- [ ] Type hints vérifiés avec mypy : `mypy arslm/`
- [ ] Documentation complète dans README.md
- [ ] CHANGELOG.md mis à jour
- [ ] Version correctement définie dans `arslm/__init__.py`

### ✅ Phase 2 : Configuration PyPI

- [ ] Compte créé sur [PyPI.org](https://pypi.org)
- [ ] Compte créé sur [TestPyPI.org](https://test.pypi.org)
- [ ] Token API généré sur PyPI
- [ ] Token API généré sur TestPyPI
- [ ] Fichier `~/.pypirc` configuré

### ✅ Phase 3 : Tests Locaux

- [ ] Package construit localement : `python -m build`
- [ ] Package vérifié : `twine check dist/*`
- [ ] Installation locale testée : `pip install dist/*.whl`
- [ ] Tests d'import réussis : `python -c "import arslm; print(arslm.__version__)"`

---

## 📝 Étape par Étape

### Étape 1 : Setup Initial

```bash
# Cloner le repo (ou créer la structure)
git clone https://github.com/benjaminpolydeq/ARSLM.git
cd ARSLM

# Créer environnement virtuel
python -m venv venv

# Activer (Linux/Mac)
source venv/bin/activate

# Activer (Windows)
venv\Scripts\activate

# Installer en mode développement
pip install -e ".[dev]"
```

### Étape 2 : Structure du Projet

Vérifier que vous avez cette structure :

```
ARSLM/
├── arslm/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── attention.py
│   │   ├── recurrent.py
│   │   └── adaptive.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── config.py
│   │   └── preprocessing.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── server.py
│   │   └── schemas.py
│   └── cli/
│       ├── __init__.py
│       └── commands.py
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_attention.py
│   └── conftest.py
├── docs/
├── examples/
├── scripts/
│   └── publish.sh
├── .github/
│   └── workflows/
│       ├── test.yml
│       └── publish.yml
├── pyproject.toml
├── setup.py
├── MANIFEST.in
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt
└── requirements-dev.txt
```

### Étape 3 : Vérification du Code

```bash
# Tests unitaires
pytest tests/ -v --cov=arslm --cov-report=html

# Format du code
black arslm/ tests/
isort arslm/ tests/

# Vérification
flake8 arslm/ tests/ --max-line-length=88 --extend-ignore=E203,W503
mypy arslm/ --ignore-missing-imports
```

### Étape 4 : Configuration PyPI

#### A. Créer les Comptes

1. **PyPI** : https://pypi.org/account/register/
2. **TestPyPI** : https://test.pypi.org/account/register/

#### B. Générer les Tokens

1. Connectez-vous à PyPI
2. Allez dans Account Settings → API tokens
3. Créez un token avec scope "Entire account" (first time) or "Project: arslm"
4. **Copiez le token immédiatement** (vous ne pourrez plus le voir)

#### C. Configurer ~/.pypirc

```bash
# Créer le fichier de configuration
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Votre token PyPI complet

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBp...  # Votre token TestPyPI complet
EOF

# Sécuriser le fichier
chmod 600 ~/.pypirc
```

### Étape 5 : Construction du Package

```bash
# Nettoyer les anciennes builds
rm -rf build/ dist/ *.egg-info

# Construire le package
python -m build

# Vérifier le contenu
ls -lh dist/

# Devrait afficher :
# arslm-0.1.0-py3-none-any.whl
# arslm-0.1.0.tar.gz
```

### Étape 6 : Vérification du Package

```bash
# Vérifier le package
twine check dist/*

# Installer localement pour tester
pip install dist/arslm-0.1.0-py3-none-any.whl

# Tester l'import
python -c "import arslm; print(arslm.__version__)"

# Tester les fonctionnalités de base
python -c "from arslm import ARSLM; model = ARSLM(); print('OK')"
```

### Étape 7 : Publication sur TestPyPI

```bash
# Upload vers TestPyPI
twine upload --repository testpypi dist/*

# Après succès, tester l'installation depuis TestPyPI
pip uninstall arslm  # Si installé localement

pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            arslm

# Vérifier
python -c "import arslm; print(arslm.__version__)"
```

### Étape 8 : Publication sur PyPI (Production)

⚠️ **ATTENTION** : Cette étape est irréversible !

```bash
# Vérification finale
echo "Version actuelle : $(python -c 'import arslm; print(arslm.__version__)')"
echo "Vérifier que c'est la bonne version !"

# Upload vers PyPI
twine upload dist/*

# En cas de succès :
# ✅ Package publié sur https://pypi.org/project/arslm/

# Installation publique
pip install arslm

# Vérification
python -c "import arslm; print(arslm.__version__)"
```

### Étape 9 : Post-Publication

```bash
# Créer un tag Git
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Créer une release sur GitHub
# Aller sur : https://github.com/benjaminpolydeq/ARSLM/releases/new
# - Tag: v0.1.0
# - Title: ARSLM v0.1.0 - Initial Release
# - Description: Copier depuis CHANGELOG.md

# Mettre à jour la documentation
# Si vous utilisez ReadTheDocs, déclencher un nouveau build
```

---

## 🚀 Publication Automatique avec Script

Pour simplifier le processus, utilisez le script fourni :

```bash
# Rendre le script exécutable
chmod +x scripts/publish.sh

# Publication sur TestPyPI
./scripts/publish.sh --test

# Publication sur PyPI (production)
./scripts/publish.sh --prod

# Sans tests (plus rapide)
./scripts/publish.sh --prod --no-test
```

---

## 🔄 Mise à Jour d'une Version

### 1. Modifier la Version

```python
# Dans arslm/__init__.py
__version__ = "0.2.0"  # Nouvelle version
```

### 2. Mettre à Jour CHANGELOG.md

```markdown
## [0.2.0] - 2025-01-XX

### Added
- Nouvelle fonctionnalité X
- Support pour Y

### Fixed
- Correction du bug Z
```

### 3. Republier

```bash
# Clean
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Check
twine check dist/*

# Upload
twine upload dist/*

# Tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

---

## ❓ Résolution de Problèmes

### Erreur : "File already exists"

```bash
# La version existe déjà sur PyPI
# Solution : Changer la version dans arslm/__init__.py
```

### Erreur : "Invalid authentication credentials"

```bash
# Token incorrect ou expiré
# Solution : Regénérer le token sur PyPI et mettre à jour ~/.pypirc
```

### Erreur : "Package validation failed"

```bash
# Problème dans le package
# Solution : Vérifier avec twine check dist/*
# Consulter les logs d'erreur
```

### Erreur d'Import après Installation

```bash
# Module non trouvé
# Solution : Vérifier la structure dans le .whl
unzip -l dist/arslm-*.whl

# Vérifier MANIFEST.in et pyproject.toml
```

---

## 📊 Checklist Post-Publication

- [ ] Package visible sur PyPI : https://pypi.org/project/arslm/
- [ ] Installation fonctionne : `pip install arslm`
- [ ] Import fonctionne : `import arslm`
- [ ] Tests passent après installation depuis PyPI
- [ ] Documentation accessible
- [ ] Badge PyPI ajouté au README
- [ ] GitHub release créée
- [ ] Tag Git poussé
- [ ] Annonce sur les réseaux sociaux / blog
- [ ] Mise à jour des exemples de code
- [ ] Notification aux beta testeurs

---

## 🎉 Félicitations !

Votre package ARSLM est maintenant disponible sur PyPI !

**Installation publique :**
```bash
pip install arslm
```

**Links :**
- PyPI : https://pypi.org/project/arslm/
- GitHub : https://github.com/benjaminpolydeq/ARSLM
- Documentation : https://arslm.readthedocs.io

---

## 📞 Support

En cas de problème :

1. Consulter la [documentation PyPI](https://packaging.python.org/)
2. Ouvrir une issue sur GitHub
3. Contacter : benjokama@hotmail.fr

---

**Bonne publication ! 🚀**
