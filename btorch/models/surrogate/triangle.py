import torch

from btorch import jit

from .base import SurrogateFunctionBase


@jit.script
def _triangle_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.clamp(1.0 - (alpha * x).abs(), min=0.0)


@jit.script
def _triangle_derivative(x: torch.Tensor, alpha: float, damping: float) -> torch.Tensor:
    v_scaled = alpha * x
    grad = (1.0 - v_scaled.abs()).clamp(min=0.0)
    return damping * grad * alpha


class Triangle(SurrogateFunctionBase):
    """Triangular surrogate gradient with optional damping."""

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _triangle_primitive(x, self.alpha)

    def derivative(self, x: torch.Tensor, damping_factor: float = 1.0) -> torch.Tensor:
        return _triangle_derivative(x, self.alpha, damping_factor)


def triangle(
    x: torch.Tensor,
    alpha: float = 1.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return Triangle(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
