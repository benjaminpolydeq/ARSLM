# ================================
# 🔹 streamlit_app.py – ARSLM SaaS Ready
# ================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================================
# ⚙️ Config
# ================================
MODEL_PATH = "./model_checkpoint"     # chemin vers ton modèle fine-tuné
TOKENIZER_PATH = "./tokenizer_checkpoint"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# 🔹 Charger modèle et tokenizer
# ================================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # Si tu as utilisé LoRA
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH.replace("_lora",""), device_map="auto")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model

tokenizer, model = load_model()

# ================================
# 🔹 Génération de réponse
# ================================
def generate_response(user_input, max_length=200, temperature=0.8, top_p=0.9):
    prompt = f"You are ARSLM, an intelligent and friendly assistant that speaks English.\nUser: {user_input}\nARSLM:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("ARSLM:")[-1].strip()
    return response

# ================================
# 🔹 Streamlit Interface
# ================================
st.title("🤖 ARSLM - MicroLLM SaaS")
st.write("ARSLM est prêt à discuter en anglais !")

user_input = st.text_input("💬 Pose une question à ARSLM :")

if st.button("Envoyer") and user_input.strip() != "":
    with st.spinner("ARSLM réfléchit..."):
        answer = generate_response(user_input)
        st.markdown(f"**ARSLM:** {answer}")