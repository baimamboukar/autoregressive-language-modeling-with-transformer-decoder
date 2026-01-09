import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """

        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """

        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)

        Handles arbitrary batch dimensions like PyTorch
        """
        # Store input for backward pass
        self.A = A

        # Store original shape for reshape back
        self.original_shape = A.shape

        # Reshape to 2D: (*, in_features) -> (batch_size, in_features)
        A_2d = A.reshape(-1, A.shape[-1])

        # Apply linear transformation: Z = A @ W.T + b
        Z_2d = A_2d @ self.W.T + self.b

        # Reshape back to original batch dimensions: (batch_size, out_features) -> (*, out_features)
        Z = Z_2d.reshape(*self.original_shape[:-1], self.W.shape[0])

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Reshape gradients to 2D for computation
        dLdZ_2d = dLdZ.reshape(-1, dLdZ.shape[-1])
        A_2d = self.A.reshape(-1, self.A.shape[-1])

        # Compute gradients
        # dL/dA = dL/dZ @ W
        dLdA_2d = dLdZ_2d @ self.W

        # dL/dW = dL/dZ.T @ A
        self.dLdW = dLdZ_2d.T @ A_2d

        # dL/db = sum over batch dimension of dL/dZ
        self.dLdb = np.sum(dLdZ_2d, axis=0)

        # Reshape gradient back to original input shape
        self.dLdA = dLdA_2d.reshape(self.original_shape)

        return self.dLdA
