import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# 1) ARSLM Cellule
# ============================
class ARSCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.candidate_mlp = nn.Sequential(
            nn.Linear(hidden_size*2 + input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size*2 + input_size, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x_embed, h_prev, h_prev2):
        context = torch.cat([h_prev, h_prev2, x_embed], dim=-1)
        candidate = self.candidate_mlp(context)
        gate = self.gate_net(context)
        residual = 0.1 * x_embed
        h_next = h_prev + gate * candidate + residual
        h_next = self.layer_norm(h_next)
        h_next = self.dropout(h_next)
        return h_next

# ============================
# 2) ARSLM Model
# ============================
class ARSLMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_size=64, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList([ARSCell(hidden_size, emb_dim) for _ in range(num_layers)])
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.head = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.emb(input_ids)
        h_prev = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        h_prev2 = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        all_hiddens = [[] for _ in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = x[:,t,:]
            for l in range(self.num_layers):
                h_next = self.layers[l](x_t, h_prev[l], h_prev2[l])
                h_prev2[l] = h_prev[l]
                h_prev[l] = h_next
                all_hiddens[l].append(h_next)
                x_t = h_next

            # Attention sur dernière couche
            last_layer_h = all_hiddens[-1][-1]
            hist = torch.stack(all_hiddens[-1], dim=1)
            scores = self.attention(hist).squeeze(-1)
            weights = F.softmax(scores, dim=1)
            context = torch.sum(weights.unsqueeze(-1) * hist, dim=1)
            attended_h = last_layer_h + context
            logits = self.head(attended_h)
            outputs.append(logits.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# ============================
# 3) Wrapper MicroLLM
# ============================
class MicroLLM:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {i:w for w,i in vocab.items()}
        self.model = ARSLMModel(vocab_size=len(vocab))
        self.device = "cpu"  # Streamlit Cloud CPU
        self.model.to(self.device)

    def encode(self, txt):
        tokens = [self.vocab.get(w,0) for w in txt.lower().split()]
        return torch.tensor([tokens], dtype=torch.long)

    def decode(self, toks):
        return " ".join(self.inv_vocab.get(int(t), "<unk>") for t in toks)

    @torch.no_grad()
    def generate(self, prompt, max_tokens=20):
        x = self.encode(prompt).to(self.device)
        logits = self.model(x)
        toks = torch.argmax(logits, dim=-1)[0]
        return self.decode(toks)

# ============================
# 4) Streamlit App
# ============================
st.set_page_config(page_title="MicroLLM Studio ARSLM", layout="wide")
st.title("💡 MicroLLM Studio ARSLM - Chat privé")

# Initialiser vocabulaire simple pour la démo
vocab = {"bonjour":1, "merci":2, "client":3, "achat":4, "service":5, "STOP":6}

# Stockage de l'instance et historique
if 'llm' not in st.session_state:
    st.session_state['llm'] = MicroLLM(vocab)
if 'history' not in st.session_state:
    st.session_state['history'] = []

llm = st.session_state['llm']

# Zone de chat
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Votre message")
    submit = st.form_submit_button("Envoyer")

if submit and user_input:
    # Génération du modèle
    response = llm.generate(user_input)
    st.session_state['history'].append(("Vous", user_input))
    st.session_state['history'].append(("IA", response))

# Affichage de l'historique
for speaker, msg in st.session_state['history']:
    if speaker == "Vous":
        st.markdown(f"**{speaker}:** {msg}")
    else:
        st.markdown(f"**{speaker}:** {msg}")
