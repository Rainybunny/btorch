"""SpikeNet-style neuron dynamics for btorch.

This module implements a neuron that follows SpikeNet simulator update order:

1. detect spikes from previous-step membrane voltage
2. reset spiking neurons and update refractory counters
3. integrate membrane voltage for non-refractory neurons

Compared to typical BaseNode dynamics (charge -> fire -> reset), this yields
one-step shifted spike timing when input drives the membrane across threshold.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...types import TensorLike
from .. import environ
from ..base import BaseNode
from ..surrogate import Sigmoid


def _normalize_neuron_model(neuron_model: str | int) -> Literal["lif", "elif"]:
    if isinstance(neuron_model, str):
        key = neuron_model.lower()
        if key in {"lif", "0"}:
            return "lif"
        if key in {"elif", "exp_lif", "1"}:
            return "elif"
    elif int(neuron_model) == 0:
        return "lif"
    elif int(neuron_model) == 1:
        return "elif"

    raise ValueError(
        "neuron_model must be one of {'lif', 'elif', 0, 1}, "
        f"got {neuron_model!r}."
    )


class SpikeNetNeuron(BaseNode):
    """SpikeNet-compatible neuron with LIF or ELIF membrane dynamics.

    The default parameters mirror SpikeNet's NeuroPop defaults in C++.

    Args:
        n_neuron: Number of neurons.
        neuron_model: "lif"/0 or "elif"/1.
        v_threshold: Spike threshold (mV).
        v_reset: Reset potential after spike (mV).
        v_lk: Leak reversal potential (mV).
        c_m: Membrane capacitance.
        g_lk: Leak conductance.
        tau_ref: Absolute refractory period (ms).
        elif_delta_t: ELIF delta-T parameter (mV).
        elif_v_t: ELIF soft threshold voltage (mV).
        spike_freq_adapt: If True, enable SpikeNet-style K adaptation.
        v_k: K reversal potential (mV).
        dg_k: K conductance increment per spike.
        tau_k: K conductance decay time constant (ms).
        trainable_param: Names of trainable parameters.
        surrogate_function: Surrogate gradient function.
        detach_reset: If True, detach reset branch from graph.
        pre_spike_v: If True, store membrane voltage before reset.
        step_mode: Step mode. Only "s" is supported.
        backend: Backend implementation. Only "torch" is supported.
        device: Tensor device.
        dtype: Tensor dtype.
    """

    ref_step_left: torch.Tensor
    i_leak: torch.Tensor
    i_input: torch.Tensor
    i_k: torch.Tensor
    g_k: torch.Tensor

    c_m: torch.Tensor | torch.nn.Parameter
    g_lk: torch.Tensor | torch.nn.Parameter
    v_lk: torch.Tensor | torch.nn.Parameter
    tau_ref: torch.Tensor | torch.nn.Parameter
    elif_delta_t: torch.Tensor | torch.nn.Parameter
    elif_v_t: torch.Tensor | torch.nn.Parameter
    v_k: torch.Tensor | torch.nn.Parameter
    dg_k: torch.Tensor | torch.nn.Parameter
    tau_k: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        neuron_model: str | int = "lif",
        v_threshold: float | Float[TensorLike, " n_neuron"] = -50.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = -60.0,
        v_lk: float | Float[TensorLike, " n_neuron"] = -70.0,
        c_m: float | Float[TensorLike, " n_neuron"] = 0.25,
        g_lk: float | Float[TensorLike, " n_neuron"] = 0.0167,
        tau_ref: float | Float[TensorLike, " n_neuron"] = 2.0,
        elif_delta_t: float | Float[TensorLike, " n_neuron"] = 0.0,
        elif_v_t: float | Float[TensorLike, " n_neuron"] = -50.0,
        spike_freq_adapt: bool = False,
        v_k: float | Float[TensorLike, " n_neuron"] = -85.0,
        dg_k: float | Float[TensorLike, " n_neuron"] = 0.01,
        tau_k: float | Float[TensorLike, " n_neuron"] = 80.0,
        trainable_param: set[str] = set(),
        surrogate_function: Callable = Sigmoid(),
        detach_reset: bool = False,
        pre_spike_v: bool = False,
        step_mode: Literal["s"] = "s",
        backend: Literal["torch"] = "torch",
        device=None,
        dtype=None,
    ):
        super().__init__(
            n_neuron=n_neuron,
            v_threshold=v_threshold,
            v_reset=v_reset,
            trainable_param=trainable_param,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            hard_reset=True,
            pre_spike_v=pre_spike_v,
            step_mode=step_mode,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        _factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.neuron_model = _normalize_neuron_model(neuron_model)
        self.spike_freq_adapt = bool(spike_freq_adapt)

        self.def_param(
            "c_m",
            c_m,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "g_lk",
            g_lk,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "v_lk",
            v_lk,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "tau_ref",
            tau_ref,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )

        self.def_param(
            "elif_delta_t",
            elif_delta_t,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "elif_v_t",
            elif_v_t,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )

        self.def_param(
            "v_k",
            v_k,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "dg_k",
            dg_k,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "tau_k",
            tau_k,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )

        self.set_reset_value("v", self.v_lk.detach().clone(), strict=False)

        self.register_memory(
            "ref_step_left",
            0,
            self.n_neuron,
            dtype=torch.int64,
        )
        self.register_memory("i_leak", 0.0, self.n_neuron, persistent=False)
        self.register_memory("i_input", 0.0, self.n_neuron, persistent=False)
        self.register_memory("i_k", 0.0, self.n_neuron, persistent=False)

        if self.spike_freq_adapt:
            self.register_memory("g_k", 0.0, self.n_neuron)

    def _refractory_steps(
        self, dt: float, shape: torch.Size
    ) -> torch.Tensor:
        ref_steps = torch.round(torch.clamp_min(self.tau_ref, 0.0) / float(dt)).to(
            torch.int64
        )
        if ref_steps.shape != shape:
            ref_steps = torch.broadcast_to(ref_steps, shape)
        return ref_steps

    def _lif_leak(self, v: torch.Tensor) -> torch.Tensor:
        return -self.g_lk * (v - self.v_lk)

    def _elif_leak(self, v: torch.Tensor) -> torch.Tensor:
        delta_t = torch.clamp(self.elif_delta_t, min=1e-6)
        exp_term = self.g_lk * delta_t * torch.exp((v - self.elif_v_t) / delta_t)
        return -self.g_lk * (v - self.v_lk) + exp_term

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        _ = x
        return None

    def neuronal_adaptation(self):
        return None

    def single_step_forward(self, x: Float[Tensor, "*batch n_neuron"]):
        dt = float(environ.get("dt"))
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")

        can_spike = self.ref_step_left == 0
        spike = self.surrogate_function(self.v - self.v_threshold)
        spike = spike * can_spike.detach().to(self.v.dtype)

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.pre_spike_v:
            self.v_pre_spike = self.v.clone()

        self.v = self.v - (self.v - self.v_reset) * spike_d

        ref_steps = self._refractory_steps(dt=dt, shape=self.v.shape)
        ref_next = torch.where(spike_d > 0, ref_steps, self.ref_step_left)
        ref_next = torch.where(ref_next > 0, ref_next - 1, ref_next)
        self.ref_step_left = ref_next.to(torch.int64)

        if self.spike_freq_adapt:
            self.g_k = self.g_k + self.dg_k * spike_d
            self.g_k = self.g_k * torch.exp(-dt / torch.clamp(self.tau_k, min=1e-6))
            self.i_k = -self.g_k * (self.v - self.v_k)
        else:
            self.i_k.zero_()

        if self.neuron_model == "lif":
            self.i_leak = self._lif_leak(self.v)
        else:
            self.i_leak = self._elif_leak(self.v)

        self.i_input = x + self.i_k
        vdot = (self.i_leak + self.i_input) / self.c_m

        non_refractory = self.ref_step_left == 0
        self.v = torch.where(non_refractory, self.v + vdot * dt, self.v)
        return spike

    def extra_repr(self):
        parts = [
            f"neuron_model={self.neuron_model}",
            f"v_lk={self._format_repr_value(self.v_lk)}",
            f"c_m={self._format_repr_value(self.c_m)}",
            f"g_lk={self._format_repr_value(self.g_lk)}",
            f"tau_ref={self._format_repr_value(self.tau_ref)}",
            f"spike_freq_adapt={self.spike_freq_adapt}",
        ]
        if self.neuron_model == "elif":
            parts.extend(
                [
                    f"elif_delta_t={self._format_repr_value(self.elif_delta_t)}",
                    f"elif_v_t={self._format_repr_value(self.elif_v_t)}",
                ]
            )
        if self.spike_freq_adapt:
            parts.extend(
                [
                    f"v_k={self._format_repr_value(self.v_k)}",
                    f"dg_k={self._format_repr_value(self.dg_k)}",
                    f"tau_k={self._format_repr_value(self.tau_k)}",
                ]
            )
        base = super().extra_repr()
        if base:
            parts.append(base)
        return ", ".join(parts)
