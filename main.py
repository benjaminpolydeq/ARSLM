# main.py pour Termux / ARSLM
# Version prête à lancer, corrections de syntaxe incluses

# Import des modules ARSLM
import sys
import os

# Ajouter le dossier ARSLM au PYTHONPATH si ce n'est pas déjà fait
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import des sous-modules
from arslm import arslm
# Importer d'autres modules selon ton usage
# from arslm import arslm_engine
# from arslm import arslm_init
# from arslm import api, cli, utils, core

def main():
    """
    Fonction principale pour tester ARSLM sur Termux.
    """
    print("ARSLM est prêt à l'utilisation !")
    # Exemple : appeler une fonction de arslm si elle existe
    # arslm.some_function()

if __name__ == "__main__":
    main()

