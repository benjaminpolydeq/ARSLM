# test_arslm_cpu.py
import sys
import os

# Ajouter le dossier parent à sys.path pour que Python trouve arslm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Essayer d'importer la classe ARSLM principale
    from arslm.arslm import ARSLM
    print("✅ ARSLM importé avec succès")
except ImportError:
    # Si ARSLM n'est pas trouvée, utiliser ARSLMCore
    from arslm.arslm import ARSLMCore as ARSLM
    print("⚠️ ARSLM non trouvé, utilisation de ARSLMCore à la place")

# Initialisation du modèle sur CPU
try:
    model = ARSLM()  # Si ARSLMCore, initialise avec defaults
    print("✅ Modèle ARSLM initialisé sur CPU")
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation du modèle: {e}")
    sys.exit(1)

# Test simple de génération de texte
prompt = "Bonjour ARSLM, peux-tu me donner un exemple de réponse ?"
try:
    if hasattr(model, "generate"):
        response = model.generate(
            prompt=prompt,
            max_length=100,
            temperature=0.7,
            include_context=False
        )
        print("\n--- Résultat génération ---")
        print(f"Prompt : {prompt}")
        print(f"Réponse : {response}")
    else:
        print("❌ Le modèle n'a pas de méthode 'generate'")
except Exception as e:
    print(f"❌ Erreur lors de la génération : {e}")