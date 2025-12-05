import streamlit as st
from ARSLM.ARSLM import ARSLM

st.set_page_config(
    page_title="ARSLM AI",
    page_icon="🤖",
    layout="centered"
)

bot = ARSLM()

st.header("🤖 ARSLM — AI MVP Ready for Investors")
st.write("Un moteur d’intelligence artificielle léger, modulaire, et extensible.")

user_input = st.text_input("💬 Posez une question au modèle")

if st.button("Envoyer"):
    response = bot.chat(user_input)
    st.success(response)

st.markdown("### 📜 Historique")
for item in bot.get_history():
    st.write(f"**Vous :** {item['user']}")
    st.write(f"**ARSLM :** {item['bot']}")
    st.write("---")

st.caption("© 2025 ARSLM • MVP Demonstration Version")
