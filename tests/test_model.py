"""
Unit tests for ARSLM model.
"""

import pytest
import torch
from arslm import ARSLM, ARSLMConfig
from arslm.core.attention import MultiHeadAttention
from arslm.core.recurrent import AdaptiveLSTM
from arslm.utils.tokenizer import ARSLMTokenizer


class TestARSLMConfig:
    """Tests for ARSLMConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ARSLMConfig()
        assert config.vocab_size == 50000
        assert config.hidden_size == 768
        assert config.num_layers == 12
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ARSLMConfig(
            vocab_size=30000,
            hidden_size=512,
            num_layers=6
        )
        assert config.vocab_size == 30000
        assert config.hidden_size == 512
        assert config.num_layers == 6
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Should raise error: hidden_size not divisible by num_heads
        with pytest.raises(AssertionError):
            ARSLMConfig(hidden_size=100, num_heads=12)
    
    def test_config_serialization(self, tmp_path):
        """Test config save/load."""
        config = ARSLMConfig(vocab_size=10000)
        config_path = tmp_path / "config.json"
        
        # Save
        config.save(config_path)
        
        # Load
        loaded_config = ARSLMConfig.load(config_path)
        assert loaded_config.vocab_size == config.vocab_size


class TestARSLM:
    """Tests for main ARSLM model."""
    
    @pytest.fixture
    def config(self):
        """Create small config for testing."""
        return ARSLMConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            max_length=64
        )
    
    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        return ARSLM(config)
    
    def test_model_initialization(self, config):
        """Test model can be initialized."""
        model = ARSLM(config)
        assert model is not None
        assert model.config.vocab_size == 1000
    
    def test_model_forward(self, model):
        """Test forward pass."""
        batch_size = 2
        seq_length = 16
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Forward pass
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_length, 1000)
    
    def test_model_with_attention_mask(self, model):
        """Test forward with attention mask."""
        batch_size = 2
        seq_length = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[:, 10:] = 0  # Mask last 6 tokens
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs['logits'].shape == (batch_size, seq_length, 1000)
    
    def test_model_with_labels(self, model):
        """Test training with labels."""
        batch_size = 2
        seq_length = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        labels = torch.randint(0, 1000, (batch_size, seq_length))
        
        outputs = model(input_ids, labels=labels)
        
        assert 'loss' in outputs
        assert outputs['loss'] is not None
        assert outputs['loss'].requires_grad
    
    def test_generate(self, model):
        """Test text generation."""
        prompt = "Hello, world!"
        response = model.generate(prompt, max_length=50)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_multiple_sequences(self, model):
        """Test generating multiple sequences."""
        prompt = "Tell me a story"
        responses = model.generate(
            prompt,
            num_return_sequences=3,
            max_length=50
        )
        
        assert isinstance(responses, list)
        assert len(responses) == 3
    
    def test_num_parameters(self, model):
        """Test parameter counting."""
        total_params = model.num_parameters()
        trainable_params = model.num_parameters(only_trainable=True)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params trainable by default
    
    def test_model_save_load(self, model, tmp_path):
        """Test model save and load."""
        save_dir = tmp_path / "model"
        
        # Save model
        model.save_pretrained(save_dir)
        
        # Check files exist
        assert (save_dir / "config.json").exists()
        assert (save_dir / "pytorch_model.bin").exists()
        
        # Load model
        loaded_model = ARSLM.from_pretrained(save_dir)
        
        assert loaded_model.config.vocab_size == model.config.vocab_size
    
    @pytest.mark.parametrize("batch_size,seq_length", [
        (1, 10),
        (4, 32),
        (8, 64),
    ])
    def test_different_batch_sizes(self, model, batch_size, seq_length):
        """Test with different batch sizes."""
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        outputs = model(input_ids)
        
        assert outputs['logits'].shape[0] == batch_size
        assert outputs['logits'].shape[1] == seq_length


class TestMultiHeadAttention:
    """Tests for attention mechanisms."""
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        batch_size = 2
        seq_length = 10
        hidden_size = 128
        num_heads = 4
        
        attention = MultiHeadAttention(hidden_size, num_heads)
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        output = attention(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)
    
    def test_attention_with_mask(self):
        """Test attention with mask."""
        batch_size = 2
        seq_length = 10
        hidden_size = 128
        num_heads = 4
        
        attention = MultiHeadAttention(hidden_size, num_heads)
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)
        
        output = attention(hidden_states, attention_mask)
        
        assert output.shape == (batch_size, seq_length, hidden_size)


class TestAdaptiveLSTM:
    """Tests for adaptive recurrent networks."""
    
    def test_lstm_forward(self):
        """Test LSTM forward pass."""
        batch_size = 2
        seq_length = 10
        input_size = 128
        hidden_size = 128
        
        lstm = AdaptiveLSTM(input_size, hidden_size)
        input_tensor = torch.randn(batch_size, seq_length, input_size)
        
        output, (h_n, c_n) = lstm(input_tensor)
        
        assert output.shape == (batch_size, seq_length, hidden_size)
        assert h_n.shape[1] == batch_size
        assert c_n.shape[1] == batch_size


class TestARSLMTokenizer:
    """Tests for tokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = ARSLMTokenizer()
        assert tokenizer.vocab_size > 0
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = ARSLMTokenizer()
        text = "Hello, world!"
        
        # Encode
        token_ids = tokenizer.encode(text)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        assert isinstance(decoded, str)
    
    def test_batch_encode(self):
        """Test batch encoding."""
        tokenizer = ARSLMTokenizer()
        texts = ["Hello", "World", "Test"]
        
        batch = tokenizer.batch_encode(texts)
        
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert batch['input_ids'].shape[0] == 3
    
    def test_build_vocab(self):
        """Test vocabulary building."""
        tokenizer = ARSLMTokenizer()
        texts = ["hello world", "hello there", "world peace"]
        
        tokenizer.build_vocab(texts, vocab_size=100)
        
        assert "hello" in tokenizer.vocab
        assert "world" in tokenizer.vocab
    
    def test_save_load(self, tmp_path):
        """Test tokenizer save/load."""
        tokenizer = ARSLMTokenizer()
        tokenizer_path = tmp_path / "tokenizer.json"
        
        # Save
        tokenizer.save(tokenizer_path)
        
        # Load
        loaded_tokenizer = ARSLMTokenizer.load(tokenizer_path)
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete pipeline."""
        # Create small config
        config = ARSLMConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4
        )
        
        # Create model
        model = ARSLM(config)
        
        # Create tokenizer
        tokenizer = ARSLMTokenizer()
        
        # Encode text
        text = "Hello, world!"
        input_ids = tokenizer.encode(text, return_tensors="pt").unsqueeze(0)
        
        # Forward pass
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape[-1] == config.vocab_size


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])