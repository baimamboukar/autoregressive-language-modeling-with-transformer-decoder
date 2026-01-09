import torch
from torch import nn
import math

'''
Specification:
- Module should add positional information to input embeddings
- Uses sinusoidal position encodings as described in "Attention Is All You Need"
- Positional encoding matrix should have shape (1, max_len, d_model)
- Even indices use sine functions, odd indices use cosine functions
- Wavelengths form geometric progression from 2π to 10000·2π
- Encoding values should be on same device as input tensor
- Should handle any sequence length up to max_len
- Should raise error if input sequence length exceeds max_len
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.

        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.

        Side Effects:
            - Initializes the positional encoding buffer 'pe'
              of shape (1, max_len, d_model) (in order to broadcast with input tensor)
        """
        # Create positional encoding matrix
        pe = torch.zeros(1, max_len, d_model)

        # Create position indices [0, 1, 2, ..., max_len-1] of shape (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Create dimension indices [0, 2, 4, ..., d_model-2] for even positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        # Apply sine to even indices (0, 2, 4, ...)
        pe[0, :, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length
        """
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Step 3: Add positional encodings to input
        # self.pe has shape (1, max_len, d_model), we slice first seq_len positions
        return x + self.pe[:, :seq_len, :]
