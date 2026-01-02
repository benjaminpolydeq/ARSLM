import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "distilgpt2"
LORA_PATH = "./arslm_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD BASE MODEL
# =========================
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
base_model.to(device)

# =========================
# LOAD LoRA
# =========================
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("✅ ARSLM + LoRA loaded successfully")

# =========================
# GENERATE FUNCTION
# =========================
def generate(text, max_new_tokens=100):
    prompt = (
        "You are ARSLM, an intelligent and friendly AI assistant.\n"
        f"User: {text}\n"
        "ARSLM:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("ARSLM:")[-1].strip()

# =========================
# TEST
# =========================
while True:
    user_input = input("\n🧑 User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    answer = generate(user_input)
    print(f"🤖 ARSLM: {answer}")