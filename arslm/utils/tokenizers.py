"""
Tokenizer for ARSLM.

Handles text tokenization, encoding, and decoding.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import torch


class ARSLMTokenizer:
    """
    Tokenizer for ARSLM model.
    
    Supports character-level, word-level, and subword tokenization.
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    MASK_TOKEN = "[MASK]"
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 512,
        tokenization_type: str = "word"  # "char", "word", or "subword"
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Vocabulary dictionary mapping tokens to IDs
            max_length: Maximum sequence length
            tokenization_type: Type of tokenization
        """
        self.max_length = max_length
        self.tokenization_type = tokenization_type
        
        if vocab is None:
            # Initialize with special tokens
            self.vocab = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.BOS_TOKEN: 2,
                self.EOS_TOKEN: 3,
                self.MASK_TOKEN: 4,
            }
        else:
            self.vocab = vocab
        
        # Create reverse vocabulary
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special token IDs
        self.pad_token_id = self.vocab[self.PAD_TOKEN]
        self.unk_token_id = self.vocab[self.UNK_TOKEN]
        self.bos_token_id = self.vocab[self.BOS_TOKEN]
        self.eos_token_id = self.vocab[self.EOS_TOKEN]
        self.mask_token_id = self.vocab[self.MASK_TOKEN]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = text.strip()
        
        if self.tokenization_type == "char":
            return list(text)
        
        elif self.tokenization_type == "word":
            # Simple word tokenization
            tokens = re.findall(r'\w+|[^\w\s]', text.lower())
            return tokens
        
        elif self.tokenization_type == "subword":
            # Placeholder for BPE/WordPiece
            # In production, use HuggingFace tokenizers
            return text.split()
        
        else:
            raise ValueError(f"Unknown tokenization type: {self.tokenization_type}")
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add [BOS] and [EOS]
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate if too long
            return_tensors: Return format ("pt" for PyTorch, None for list)
            
        Returns:
            Token IDs as list or tensor
        """
        max_length = max_length or self.max_length
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Convert to IDs
        token_ids = [
            self.vocab.get(token, self.unk_token_id)
            for token in tokens
        ]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        # Truncate if necessary
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if add_special_tokens:
                token_ids[-1] = self.eos_token_id
        
        # Pad if necessary
        if padding and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        # Convert to tensor if requested
        if return_tensors == "pt":
            return torch.tensor(token_ids, dtype=torch.long)
        
        return token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            
            # Skip special tokens if requested
            if skip_special_tokens and token in [
                self.PAD_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.MASK_TOKEN
            ]:
                continue
            
            tokens.append(token)
        
        # Join tokens
        if self.tokenization_type == "char":
            return "".join(tokens)
        else:
            return " ".join(tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of input texts
            **kwargs: Arguments passed to encode()
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        kwargs['return_tensors'] = 'pt'
        
        # Encode all texts
        all_token_ids = []
        for text in texts:
            token_ids = self.encode(text, **kwargs)
            all_token_ids.append(token_ids)
        
        # Stack into batch
        input_ids = torch.stack(all_token_ids)
        
        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def batch_decode(
        self,
        token_ids: torch.Tensor,
        **kwargs
    ) -> List[str]:
        """
        Batch decode multiple sequences.
        
        Args:
            token_ids: Batch of token IDs [batch_size, seq_length]
            **kwargs: Arguments passed to decode()
            
        Returns:
            List of decoded texts
        """
        return [
            self.decode(seq, **kwargs)
            for seq in token_ids
        ]
    
    def build_vocab(
        self,
        texts: List[str],
        vocab_size: int = 10000,
        min_frequency: int = 2
    ) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of texts to build vocab from
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum token frequency
        """
        from collections import Counter
        
        # Tokenize all texts
        token_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        # Keep special tokens
        new_vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
            self.MASK_TOKEN: 4,
        }
        
        # Add most common tokens
        next_id = len(new_vocab)
        for token, count in token_counts.most_common(vocab_size):
            if count >= min_frequency and token not in new_vocab:
                new_vocab[token] = next_id
                next_id += 1
                
                if len(new_vocab) >= vocab_size:
                    break
        
        self.vocab = new_vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'vocab': self.vocab,
            'max_length': self.max_length,
            'tokenization_type': self.tokenization_type,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ARSLMTokenizer":
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(**config)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)