from collections.abc import Callable
from typing import Any, Literal

import torch
from jaxtyping import Float
from spikingjelly.activation_based import surrogate
from torch import Tensor

from .. import environ
from ..base import BaseNode
from ..ode import euler_step
from ..scale import SupportScaleState
from ..types import TensorLike


class Izhikevich(BaseNode, SupportScaleState):
    """Izhikevich neuron with quadratic membrane dynamics and recovery
    variable."""

    u: torch.Tensor

    c_m: torch.Tensor | torch.nn.Parameter
    k: torch.Tensor | torch.nn.Parameter
    v_T: torch.Tensor | torch.nn.Parameter
    a: torch.Tensor | torch.nn.Parameter
    b: torch.Tensor | torch.nn.Parameter
    d: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int,
        v_threshold: float | Float[TensorLike, "{self.n_neuron}"] = 30.0,
        v_reset: float | Float[TensorLike, "{self.n_neuron}"] = -65.0,
        v_rest: float | Float[TensorLike, "{self.n_neuron}"] = -65.0,
        c_m: float | Float[TensorLike, "{self.n_neuron}"] = 100.0,
        k: float | Float[TensorLike, "{self.n_neuron}"] = 0.7,
        v_T: float | Float[TensorLike, "{self.n_neuron}"] = -40.0,
        a: float | Float[TensorLike, "{self.n_neuron}"] = 0.03,
        b: float | Float[TensorLike, "{self.n_neuron}"] = -2.0,
        d: float | Float[TensorLike, "{self.n_neuron}"] = 100.0,
        u_init: float | Float[TensorLike, "{self.n_neuron}"] = 0.0,
        trainable_param: set[str] = set(),
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        hard_reset: bool = False,
        pre_spike_v: bool = False,
        step_mode: Literal["s"] = "s",
        backend: Literal["torch"] = "torch",
        device=None,
        dtype=None,
    ):
        super().__init__(
            n_neuron=n_neuron,
            v_threshold=v_threshold,
            v_reset=v_reset if v_rest is None else v_rest,
            trainable_param=trainable_param,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            hard_reset=hard_reset,
            pre_spike_v=pre_spike_v,
            step_mode=step_mode,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        _factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self._def_param("c_m", c_m, **_factory_kwargs)
        self._def_param("k", k, **_factory_kwargs)
        self._def_param("v_T", v_T, **_factory_kwargs)
        self._def_param("a", a, **_factory_kwargs)
        self._def_param("b", b, **_factory_kwargs)
        self._def_param("d", d, **_factory_kwargs)

        if v_rest is not None:
            self._def_param("_v_rest", v_rest, **_factory_kwargs)
        else:
            self._v_rest = None

        self.register_memory("u", u_init, self.n_neuron)

    @property
    def v_rest(self):
        if self._v_rest is None:
            return self.v_reset
        return self._v_rest

    @v_rest.setter
    def v_rest(self, v_rest):
        if self._v_rest is not None:
            self._v_rest = v_rest

    def dV(
        self,
        v: Float[Tensor, "*batch {self.n_neuron}"],
        u: Float[Tensor, "*batch {self.n_neuron}"],
        x: Float[Tensor, "*batch {self.n_neuron}"],
    ):
        quadratic = self.k * (v - self.v_rest) * (v - self.v_T)
        return (x + quadratic - u) / self.c_m

    def dU(
        self,
        u: Float[Tensor, "*batch {self.n_neuron}"],
        v: Float[Tensor, "*batch {self.n_neuron}"],
    ):
        return self.a * (self.b * (v - self.v_rest) - u)

    def neuronal_charge(self, x: Float[Tensor, "*batch {self.n_neuron}"]):
        dt = environ.get("dt")
        self.v = euler_step(self.dV, self.v, self.u, x, dt=dt)

    def neuronal_adaptation(self):
        dt = environ.get("dt")
        self.u = euler_step(self.dU, self.u, self.v, dt=dt)

    def neuronal_fire(self):
        spike = self.surrogate_function(
            (self.v - self.v_threshold) / (self.v_threshold - self.v_reset)
        )
        return spike

    def neuronal_reset(self, spike: Float[Tensor, "*batch n"]):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.pre_spike_v:
            self.v_pre_spike = self.v.clone()

        if self.hard_reset:
            self.v -= (self.v - self.v_reset) * spike_d
        else:
            self.v -= (self.v_threshold - self.v_reset) * spike_d

        self.u += self.d * spike_d

    def extra_repr(self):
        return super().extra_repr()


class IzhikevichNode(Izhikevich):
    """Backward-compatible alias for the updated Izhikevich implementation."""

    pass
