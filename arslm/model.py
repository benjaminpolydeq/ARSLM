"""
ARSLM — Adaptive Recurrent State Language Model (from-scratch).

A lightweight hybrid architecture combining adaptive recurrent blocks
and attention mechanisms for sequential language modeling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arslm.config import ARSLMConfig
from arslm.core.attention import MultiHeadAttention, AdaptiveAttention, LocalAttention
from arslm.core.recurrent import AdaptiveLSTM, AdaptiveGRU, AdaptiveRNN


class ARSLMBlock(nn.Module):
    """One hybrid layer: Recurrent → Attention → FFN with residuals."""

    def __init__(self, config: ARSLMConfig):
        super().__init__()
        self.config = config

        # Recurrent component
        recurrent_cls = {
            "lstm": AdaptiveLSTM,
            "gru": AdaptiveGRU,
            "rnn": AdaptiveRNN,
        }.get(config.recurrent_type.lower(), AdaptiveLSTM)

        self.recurrent = recurrent_cls(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            dropout=config.hidden_dropout,
            bidirectional=config.bidirectional,
        )

        # Attention component
        if config.use_adaptive_attention:
            self.attention = AdaptiveAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
            )
        elif config.local_attention_window is not None:
            self.attention = LocalAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                window_size=config.local_attention_window,
                dropout=config.attention_dropout,
            )
        else:
            self.attention = MultiHeadAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
            )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout),
        )

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state=None,
    ) -> Tuple[torch.Tensor, object]:
        # 1. Recurrent path
        residual = hidden_states
        recurrent_out, new_state = self.recurrent(hidden_states, recurrent_state)

        # Handle possible bidirectional size mismatch
        if recurrent_out.size(-1) != residual.size(-1):
            # Simple projection if needed (e.g. bidirectional)
            if not hasattr(self, "_recurrent_proj"):
                self._recurrent_proj = nn.Linear(
                    recurrent_out.size(-1), residual.size(-1), device=recurrent_out.device
                )
            recurrent_out = self._recurrent_proj(recurrent_out)

        hidden_states = self.norm1(residual + recurrent_out)

        # 2. Attention path
        residual = hidden_states
        if isinstance(self.attention, MultiHeadAttention):
            attn_out = self.attention(hidden_states, attention_mask)
        else:
            attn_out = self.attention(hidden_states, attention_mask)
        hidden_states = self.norm2(residual + attn_out)

        # 3. FFN
        residual = hidden_states
        hidden_states = self.norm3(residual + self.ffn(hidden_states))

        return hidden_states, new_state


class ARSLM(nn.Module):
    """
    Adaptive Recurrent State Language Model.

    A from-scratch hybrid language model that interleaves adaptive
    recurrent units and attention for efficient sequence modeling.
    """

    def __init__(self, config: ARSLMConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout)

        self.layers = nn.ModuleList(
            [ARSLMBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)
        hidden_states = self.embed_dropout(hidden_states)

        # Stack of hybrid blocks
        recurrent_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            hidden_states, recurrent_states[i] = layer(
                hidden_states, attention_mask, recurrent_states[i]
            )

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        eos_token_id = eos_token_id or self.config.eos_token_id
        generated = input_ids

        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            next_token_logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_positions.weight.numel()
        return n
