# streamlit_app.py
import streamlit as st
from microllm_core import MicroLLMCore
from datetime import datetime

st.set_page_config(page_title="MicroLLM Studio", layout="wide")

# --- Sidebar / Configuration
with st.sidebar:
    st.header("MicroLLM Studio")
    st.write("Prototype | Démo légère")
    mode = st.radio("Mode", ["MVP (léger)", "Studio (UI complète)"])
    model_name = st.selectbox("Modèle", ["MicroLLM-lite"], index=0)
    show_investor = st.checkbox("Afficher la proposition investisseur", value=False)
    st.markdown("---")
    st.caption("Déploye sur Streamlit Cloud : repository `TON_PSEUDO/micro-llm`, main file `streamlit_app.py`")

# --- Initialize core and session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "text": "Bienvenue sur MicroLLM Studio — demo légère."}
    ]

core = MicroLLMCore(name=model_name)

# --- Main layout
col1, col2 = st.columns([3, 1]) if mode == "Studio (UI complète)" else st.columns([1, 1])

with col1:
    st.title("💬 MicroLLM Chat")
    st.write("Un prototype de chat léger. Cette version n'utilise pas de modèles lourds (demo).")
    # Message display
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        txt = msg.get("text", "")
        ts = msg.get("time", "")
        if role == "user":
            st.markdown(f"**Vous** • {ts}")
            st.info(txt)
        elif role == "assistant":
            st.markdown(f"**MicroLLM** • {ts}")
            st.success(txt)
        else:
            st.markdown(f"_{txt}_")

    # Input
    user_input = st.text_area("Écris un message", height=120)
    col_send, col_clear = st.columns([1,1])
    with col_send:
        if st.button("Envoyer"):
            if user_input.strip():
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state.messages.append({"role":"user","text":user_input,"time":now})
                # generate reply
                out = core.generate_reply(user_input, st.session_state.messages)
                now2 = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state.messages.append({"role":"assistant","text":out["text"], "time":now2, "meta":out["meta"]})
                # clear input by rerun
                st.experimental_rerun()
    with col_clear:
        if st.button("Effacer la conversation"):
            st.session_state.messages = [{"role":"system","text":"Conversation réinitialisée."}]
            st.experimental_rerun()

with col2:
    st.header("🛠️ Contrôles")
    st.write(f"Mode: **{mode}**")
    st.write(f"Model: **{model_name}**")
    if st.session_state.messages:
        last = st.session_state.messages[-1]
        if last.get("meta"):
            st.markdown("**Dernière réponse — méta**")
            st.json(last["meta"])
    if show_investor:
        st.markdown("---")
        st.header("Proposition pour investisseurs")
        try:
            with open("INVESTOR_PROPOSAL.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception:
            st.info("Fichier INVESTOR_PROPOSAL.md non trouvé dans le repo.")

# Footer
st.markdown("---")
st.caption("MicroLLM — Prototype. Pour production, connecter un vrai modèle (transformers/torch) et sécuriser l'accès.")