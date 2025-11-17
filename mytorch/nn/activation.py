import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        # Numerically stable softmax: subtract max for stability
        Z_max = np.max(Z, axis=self.dim, keepdims=True)
        Z_stable = Z - Z_max

        # Compute exponentials
        exp_Z = np.exp(Z_stable)

        # Compute softmax: exp(Z) / sum(exp(Z)) along specified dimension
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)
        self.A = exp_Z / sum_exp_Z

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # For softmax: dL/dZ_i = A_i * (dL/dA_i - sum_j(A_j * dL/dA_j))
        # This is the Jacobian of softmax applied efficiently

        # Compute sum of A_j * dL/dA_j along the softmax dimension
        sum_term = np.sum(self.A * dLdA, axis=self.dim, keepdims=True)

        # Apply the softmax gradient formula
        dLdZ = self.A * (dLdA - sum_term)

        return dLdZ
 

    