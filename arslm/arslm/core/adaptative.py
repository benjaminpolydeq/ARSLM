"""
Adaptive Components for ARSLM.

Implements dynamic adaptation mechanisms that allow the model to adjust
its behavior based on input characteristics and context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class AdaptiveLayer(nn.Module):
    """
    Adaptive layer that dynamically adjusts processing based on input.
    
    Uses a gating mechanism to control information flow.
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_experts: int = 4
    ):
        """
        Initialize Adaptive Layer.
        
        Args:
            hidden_size: Dimension of hidden states
            dropout: Dropout probability
            num_experts: Number of expert networks
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size)
            )
            for _ in range(num_experts)
        ])
        
        # Gating network (router)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_experts)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptive layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            
        Returns:
            Adapted output [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Compute gating weights
        # Pool over sequence dimension for routing decision
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        gate_logits = self.gate(pooled)  # [batch_size, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(hidden_states)  # [batch_size, seq_length, hidden_size]
            expert_outputs.append(output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq, hidden, experts]
        
        # Weighted combination of expert outputs
        gate_weights = gate_weights.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, experts]
        output = (expert_outputs * gate_weights).sum(dim=-1)  # [batch, seq, hidden]
        
        output = self.dropout(output)
        
        return output


class DynamicRouter(nn.Module):
    """
    Dynamic routing mechanism for adaptive computation.
    
    Routes inputs to different processing paths based on learned criteria.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_paths: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize Dynamic Router.
        
        Args:
            hidden_size: Dimension of hidden states
            num_paths: Number of routing paths
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_paths = num_paths
        
        # Routing decision network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_paths)
        )
        
        # Processing paths
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )
            for _ in range(num_paths)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_routing_weights: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with dynamic routing.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            return_routing_weights: Whether to return routing weights
            
        Returns:
            Routed output [batch_size, seq_length, hidden_size]
        """
        # Compute routing weights
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        routing_logits = self.router(pooled)  # [batch_size, num_paths]
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Apply each path
        path_outputs = []
        for path in self.paths:
            output = path(hidden_states)
            path_outputs.append(output)
        
        # Stack and combine
        path_outputs = torch.stack(path_outputs, dim=-1)  # [batch, seq, hidden, paths]
        routing_weights = routing_weights.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, paths]
        
        output = (path_outputs * routing_weights).sum(dim=-1)
        output = self.dropout(output)
        
        if return_routing_weights:
            return output, routing_weights.squeeze()
        return output


class AdaptiveComputationTime(nn.Module):
    """
    Adaptive Computation Time mechanism.
    
    Allows the model to perform variable amounts of computation per token.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_steps: int = 10,
        threshold: float = 0.99,
        dropout: float = 0.1
    ):
        """
        Initialize ACT mechanism.
        
        Args:
            hidden_size: Dimension of hidden states
            max_steps: Maximum computation steps
            threshold: Halting threshold
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.threshold = threshold
        
        # Processing unit
        self.processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Halting probability predictor
        self.halting_predictor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive computation.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            
        Returns:
            - Output tensor [batch_size, seq_length, hidden_size]
            - Ponder cost (average computation steps)
        """
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize
        state = hidden_states
        halting_prob = torch.zeros(batch_size, seq_length, 1, device=device)
        remainders = torch.zeros(batch_size, seq_length, 1, device=device)
        n_updates = torch.zeros(batch_size, seq_length, 1, device=device)
        accumulated_state = torch.zeros_like(hidden_states)
        
        # Iterative processing
        for step in range(self.max_steps):
            # Process current state
            processed = self.processor(state)
            
            # Predict halting probability
            p = self.halting_predictor(processed)
            
            # Update accumulated probability
            still_running = (halting_prob < self.threshold).float()
            new_halting_prob = halting_prob + p * still_running
            
            # Compute update weights
            update_weights = p * still_running
            update_weights = torch.where(
                new_halting_prob > self.threshold,
                1.0 - halting_prob,
                update_weights
            )
            
            # Accumulate state
            accumulated_state = accumulated_state + processed * update_weights
            
            # Update tracking variables
            halting_prob = new_halting_prob
            n_updates = n_updates + still_running
            
            # Check if all sequences have halted
            if (halting_prob >= self.threshold).all():
                break
            
            # Update state for next iteration
            state = processed
        
        # Compute ponder cost (average number of steps)
        ponder_cost = n_updates.mean()
        
        output = self.dropout(accumulated_state)
        
        return output, ponder_cost


class AdaptiveNormalization(nn.Module):
    """
    Adaptive normalization that adjusts based on input statistics.
    
    Combines benefits of LayerNorm and BatchNorm adaptively.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5
    ):
        """
        Initialize Adaptive Normalization.
        
        Args:
            hidden_size: Dimension of hidden states
            eps: Epsilon for numerical stability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        
        # Adaptive gating between layer and batch norm
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptive normalization.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            
        Returns:
            Normalized output
        """
        # Layer normalization
        layer_mean = hidden_states.mean(dim=-1, keepdim=True)
        layer_var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        layer_norm = (hidden_states - layer_mean) / torch.sqrt(layer_var + self.eps)
        
        # Batch normalization
        batch_mean = hidden_states.mean(dim=(0, 1), keepdim=True)
        batch_var = hidden_states.var(dim=(0, 1), keepdim=True, unbiased=False)
        batch_norm = (hidden_states - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        # Compute gating weight
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        gate = self.gate_network(pooled).unsqueeze(1)  # [batch_size, 1, 1]
        
        # Adaptive combination
        normalized = gate * layer_norm + (1 - gate) * batch_norm
        
        # Apply affine transformation
        output = self.gamma * normalized + self.beta
        
        return output


class ContextGating(nn.Module):
    """
    Context-based gating mechanism for selective information flow.
    """
    
    def __init__(
        self,
        hidden_size: int,
        context_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize Context Gating.
        
        Args:
            hidden_size: Dimension of hidden states
            context_size: Dimension of context (defaults to hidden_size)
            dropout: Dropout probability
        """
        super().__init__()
        
        context_size = context_size or hidden_size
        
        # Gate computation
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + context_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply context-based gating.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            context: Context tensor [batch_size, seq_length, context_size]
            
        Returns:
            Gated output
        """
        # Concatenate hidden states and context
        combined = torch.cat([hidden_states, context], dim=-1)
        
        # Compute gate
        gate = self.gate(combined)
        
        # Apply gate
        output = hidden_states * gate
        output = self.dropout(output)
        
        return output