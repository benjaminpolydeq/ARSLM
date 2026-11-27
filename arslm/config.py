"""Configuration for the ARSLM from-scratch model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ARSLMConfig:
    """Hyperparameters for the Adaptive Recurrent State Language Model."""

    # Vocabulary & embeddings
    vocab_size: int = 32000
    hidden_size: int = 512
    max_position_embeddings: int = 2048

    # Architecture depth
    num_layers: int = 6
    num_attention_heads: int = 8

    # Recurrent settings
    recurrent_type: str = "lstm"  # "lstm" | "gru" | "rnn"
    bidirectional: bool = False

    # Attention settings
    attention_dropout: float = 0.1
    use_adaptive_attention: bool = True
    local_attention_window: Optional[int] = None  # None = full attention

    # Feed-forward / regularization
    intermediate_size: int = 2048
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Generation defaults
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Misc
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
