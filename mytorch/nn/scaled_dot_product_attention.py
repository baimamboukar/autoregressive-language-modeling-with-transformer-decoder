import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)  # Apply softmax along the last dimension (S dimension)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # Store inputs for backward pass
        self.Q = Q
        self.K = K
        self.V = V

        # Get the dimension for scaling
        d_k = Q.shape[-1]  # E dimension

        # Calculate attention scores: Q @ K.T / sqrt(d_k)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        scores = Q @ np.swapaxes(K, -2, -1)  # Transpose last two dimensions of K
        scaled_dot_product = scores / np.sqrt(d_k)

        # Apply mask before softmax if provided
        # If mask is not None, subtract self.eps from masked positions to make them -inf
        if mask is not None:
            scaled_dot_product = scaled_dot_product - self.eps * mask

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev)
        output = self.attention_scores @ V

        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # Calculate gradients for V: (N, ..., H, S, Ev)
        # d_output: (N, ..., H, L, Ev) @ attention_scores^T: (N, ..., H, S, L) -> (N, ..., H, S, Ev)
        d_V = np.swapaxes(self.attention_scores, -2, -1) @ d_output

        # Calculate gradients for attention scores: (N, ..., H, L, S)
        # d_output: (N, ..., H, L, Ev) @ V^T: (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        d_attention_scores = d_output @ np.swapaxes(self.V, -2, -1)

        # Backpropagate through softmax
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)

        # Scale gradients by 1/sqrt(d_k) (from the forward scaling)
        d_k = self.Q.shape[-1]
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(d_k)

        # Calculate gradients for Q: (N, ..., H, L, E)
        # d_scaled_dot_product: (N, ..., H, L, S) @ K: (N, ..., H, S, E) -> (N, ..., H, L, E)
        d_Q = d_scaled_dot_product @ self.K

        # Calculate gradients for K: (N, ..., H, S, E)
        # d_scaled_dot_product^T: (N, ..., H, S, L) @ Q: (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.swapaxes(d_scaled_dot_product, -2, -1) @ self.Q

        return d_Q, d_K, d_V

