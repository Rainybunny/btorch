import torch

from btorch import jit

from .base import SurrogateFunctionBase


_sigmoid_primitive = jit.script(lambda x, alpha: torch.sigmoid(alpha * x))
_sigmoid_derivative = jit.script(
    lambda x, alpha: alpha * torch.sigmoid(alpha * x) * (1 - torch.sigmoid(alpha * x))
)


class Sigmoid(SurrogateFunctionBase):
    """Logistic surrogate derivative."""

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _sigmoid_primitive(x, self.alpha)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return _sigmoid_derivative(x, self.alpha)
