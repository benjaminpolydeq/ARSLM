# arslm/arslm.py
from arslm.core.engine import ARSLMEngine

class ARSLM:
    """
    Wrapper pour ARSLMEngine compatible avec Streamlit_app.py
    Fournit generate(prompt, ...) et clear_history()
    """

    def __init__(self, device="cpu", custom_model=False, dataset_path=None):
        """
        Initialise ARSLM.
        Arguments optionnels :
        - device: 'cpu' ou 'cuda' (non utilisé ici mais pour compatibilité future)
        - custom_model: True/False (non utilisé pour SAFE MODE)
        - dataset_path: chemin vers le dataset JSON
        """
        self.engine = ARSLMEngine(dataset_path=dataset_path)

    def generate(self, prompt, max_length=150, temperature=0.7, include_context=True):
        """
        Génère une réponse à partir d'un prompt.
        Arguments max_length, temperature, include_context sont ignorés
        mais gardés pour compatibilité avec Streamlit_app.py
        """
        return self.engine.generate_response(prompt)

    def clear_history(self):
        """
        Efface l'historique de conversation
        """
        self.engine.clear_history()