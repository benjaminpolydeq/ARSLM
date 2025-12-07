import streamlit as st

st.set_page_config(page_title="MicroLLM", layout="wide")

st.title("🚀 MicroLLM — MVP en ligne")
st.write("""
Bienvenue sur MicroLLM, une plateforme légère de modèles linguistiques privés.
Ce MVP est fonctionnel et ne dépend d'aucun modèle lourd comme Torch.
""")

# Démo simple d'interaction
user_text = st.text_input("💬 Pose une question ou écris un texte")

if user_text:
    st.write("### Réponse MicroLLM")
    st.write(f"🔹 Votre texte contient **{len(user_text.split())}** mots.")
    st.write("🔹 Cette appli Streamlit fonctionne correctement ✔️")