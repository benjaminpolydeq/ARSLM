from arslm.arslm import ARSLM

arslm = ARSLM(
    use_custom_model=True,   # CRUCIAL
    device="cpu"             # CRUCIAL
)

response = arslm.generate(
    prompt="Bonjour ARSLM, test CPU Termux",
    max_length=80,
    temperature=0.7,
    include_context=False
)

print("✅ Réponse ARSLM :")
print(response)
