import streamlit as st
from ARSLM import ARSLM

# Configuration de la page
st.set_page_config(page_title="ARSLM Prototype", page_icon="🤖", layout="centered")
st.title("🤖 ARSLM — Prototype LLM")
st.markdown("Testez le modèle ARSLM avec du texte ou un fichier `.txt`")

# Initialisation du modèle
model = ARSLM()

# Sidebar pour options d'entrée
st.sidebar.header("Options d'entrée")
input_method = st.sidebar.radio("Mode d'entrée", ["Texte direct", "Fichier .txt"])

text = ""

# Choix de l'entrée
if input_method == "Texte direct":
    text = st.text_area("Entrez votre texte ici", height=200)
else:
    uploaded_file = st.file_uploader("Uploader un fichier `.txt`", type=["txt"])
    if uploaded_file is not None:
        try:
            text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")

# Bouton de prédiction
if st.button("Prédire"):
    if text.strip() == "":
        st.warning("Aucun texte fourni.")
    else:
        try:
            result = model.predict([text])
            st.success("Résultat du modèle :")
            st.write(result)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# Footer
st.markdown("---")
st.markdown("ARSLM Prototype — Développé par Benjamin Kama")

#Add final version of main.py for Streamlit

