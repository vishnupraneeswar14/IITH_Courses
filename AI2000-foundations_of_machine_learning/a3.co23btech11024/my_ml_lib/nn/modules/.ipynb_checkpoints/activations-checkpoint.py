# In my_ml_lib/nn/modules/activations.py
from .base import Module
from ..autograd import Value
# Assuming Value class has .relu() and .sigmoid() methods implemented

class ReLU(Module):
    """Applies the Rectified Linear Unit function element-wise."""

    def __call__(self, x):
        """
        Forward pass: Applies ReLU activation.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output Value object (result of x.relu()).
        """
        # TODO: Call the relu method on the input Value object
        other = x if isinstance(x, Value) else Value(x)
        return other.relu() # Placeholder

    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    """Applies the Sigmoid function element-wise."""

    def __call__(self, x):
        """
        Forward pass: Applies Sigmoid activation.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output Value object (result of x.sigmoid()).
        """
        
        other = -x if isinstance(x, Value) else Value(-x)
        return 1/(1 + other.exp()) # Placeholder

    def __repr__(self):
        return "Sigmoid()"