from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from btorch.models import base, environ


def _unflatten_td(seq: Tensor, rest_shape: Sequence[int]) -> Tensor:
    """Convert [D, T] to [T, *rest_shape] where D = prod(rest_shape)."""
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D [D, T] tensor, got {seq.shape}.")
    t = seq.shape[1]
    return seq.transpose(0, 1).reshape((t,) + rest_shape)


def randn_like(
    like: Tensor, generator: torch.Generator | None = None, **kwargs
) -> Tensor:
    return torch.empty_like(like).normal_(generator=generator, **kwargs)


def ou_noise(
    *size: int,
    sigma: Tensor,
    tau: Tensor,
    T: int,
    dt: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    noise0: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Functional OU noise generator.

    Args:
        size: Output noise shape, like torch.randn.
        sigma: Scalar or per-neuron sigma, broadcastable to noise.
        tau: Scalar or per-neuron tau, broadcastable to noise.
        T: Sequence length.
        dt: Time step.
        device: Device for generated noise0 if not provided.
        dtype: Dtype for generated noise0 if not provided.
        noise0: Optional initial noise state, shape `size`.
        generator: Optional RNG generator for noise sampling.
    """
    if noise0 is None:
        if len(size) == 0:
            raise ValueError("Provide size or noise0.")
        noise0 = torch.randn(size, device=device, dtype=dtype, generator=generator)
    elif len(size) != 0:
        if tuple(size) != tuple(noise0.shape):
            raise ValueError(f"size={size} does not match noise0.shape={noise0.shape}.")

    alpha = torch.exp(-dt / tau)
    beta = sigma * torch.sqrt(1.0 - torch.exp(-2.0 * dt / tau))

    rest_shape = noise0.shape
    D = noise0.numel()

    device = noise0.device
    dtype = noise0.dtype

    noise0_flat = noise0.reshape(-1)
    eps = torch.randn((D, T), device=device, dtype=dtype, generator=generator)

    beta_flat = beta.reshape(-1)
    if beta_flat.numel() == 1:
        beta_flat = beta_flat.expand(D)
    elif beta_flat.numel() != D:
        msg = (
            f"beta has {beta_flat.numel()} elems but D={D}; "
            "sigma must be broadcastable to noise."
        )
        raise RuntimeError(msg)

    u = beta_flat[:, None] * eps  # [D,T]

    if alpha.numel() == 1:
        a = alpha.reshape(1)

        powers = torch.arange(T - 1, -1, -1, device=device, dtype=dtype)
        w = torch.pow(a.to(dtype=dtype), powers).view(1, 1, T)  # [1,1,T]

        u_pad = F.pad(u.unsqueeze(1), (T - 1, 0))
        n_from_u = F.conv1d(u_pad, w).squeeze(1)

        factors = torch.pow(
            a.to(dtype=dtype),
            torch.arange(1, T + 1, device=device, dtype=dtype),
        )
        n_seq = n_from_u + noise0_flat[:, None] * factors[None, :]

    else:
        alpha_flat = alpha.reshape(-1)
        if alpha_flat.numel() == 1:
            alpha_flat = alpha_flat.expand(D)
        elif alpha_flat.numel() != D:
            msg = (
                f"alpha has {alpha_flat.numel()} elems but D={D}; "
                "tau must be broadcastable to noise."
            )
            raise RuntimeError(msg)

        u_ch = u.unsqueeze(0)
        u_pad = F.pad(u_ch, (T - 1, 0))

        powers = torch.arange(T - 1, -1, -1, device=device, dtype=dtype)
        w = torch.pow(alpha_flat[:, None], powers[None, :])  # [D,T]
        w = w[:, None, :]

        n_from_u = F.conv1d(u_pad, w, groups=D).squeeze(0)

        tpow = torch.arange(1, T + 1, device=device, dtype=dtype)
        factors = torch.pow(alpha_flat[:, None], tpow[None, :])
        n_seq = n_from_u + noise0_flat[:, None] * factors

    return _unflatten_td(n_seq, rest_shape)


def ou_noise_like(
    like: Tensor,
    sigma: Tensor,
    tau: Tensor,
    *,
    T: int,
    dt: float,
    noise0: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """OU noise convenience wrapper using a reference tensor.

    Args:
        like: Reference tensor providing shape, device, dtype.
        sigma: Scalar or per-neuron sigma, broadcastable to `like`.
        tau: Scalar or per-neuron tau, broadcastable to `like`.
        T: Sequence length.
        dt: Time step.
        noise0: Optional initial state. Defaults to randn_like(like).
        generator: Optional RNG generator for noise sampling.
    """
    if noise0 is None:
        noise0 = randn_like(like, generator=generator)
    return ou_noise(
        sigma=sigma,
        tau=tau,
        T=T,
        dt=dt,
        noise0=noise0,
        generator=generator,
    )


def poisson_noise(
    *size: int,
    rate: float | Tensor,
    T: int,
    dt: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Functional Poisson noise generator.

    The rate parameter is interpreted per unit time; samples are drawn
    with lambda = rate * dt for each step.
    """
    if len(size) == 0:
        raise ValueError("Provide output size for poisson_noise.")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}.")

    if dtype is None:
        sample_dtype = torch.get_default_dtype()
    else:
        sample_dtype = dtype
    if not torch.empty((), dtype=sample_dtype).is_floating_point():
        raise ValueError("poisson_noise requires a floating dtype.")

    base = torch.zeros(size, device=device, dtype=sample_dtype)
    lam = torch.as_tensor(rate, device=base.device, dtype=sample_dtype) * float(dt)
    if torch.any(lam < 0):
        raise ValueError("Poisson rate * dt must be non-negative.")

    lam_full = base + lam
    lam_seq = lam_full.unsqueeze(0).expand((T,) + tuple(size))
    return torch.poisson(lam_seq, generator=generator)


def poisson_noise_like(
    like: Tensor,
    rate: float | Tensor,
    *,
    T: int,
    dt: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Poisson noise convenience wrapper using a reference tensor."""
    dtype = like.dtype if like.is_floating_point() else torch.get_default_dtype()
    return poisson_noise(
        *like.shape,
        rate=rate,
        T=T,
        dt=dt,
        device=like.device,
        dtype=dtype,
        generator=generator,
    )


def _pink_fir_kernel(
    fir_order: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Build a deterministic FIR kernel for 1/f-like PSD.

    Uses truncated fractional-integration coefficients:
        h[0] = 1
        h[k] = h[k-1] * (k - 1 + alpha) / k,  alpha=0.5
    This gives |H(f)| ~ f^-alpha at low frequency, so PSD ~ 1/f for alpha=0.5.
    """
    if fir_order < 1:
        raise ValueError(f"fir_order must be >= 1, got {fir_order}.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("Pink-noise FIR kernel requires floating dtype.")

    alpha = 0.5
    kernel = torch.empty(fir_order, device=device, dtype=dtype)
    kernel[0] = 1.0
    for k in range(1, fir_order):
        kernel[k] = kernel[k - 1] * ((k - 1 + alpha) / k)

    kernel = kernel / kernel.norm(p=2).clamp_min(torch.finfo(dtype).eps)
    return kernel


def _white_noise_2d(
    template: Tensor, T: int, generator: torch.Generator | None = None
) -> Tensor:
    """Generate white noise with shape [D, T] matching template metadata."""
    D = template.numel()
    if generator is None:
        return torch.randn((D, T), device=template.device, dtype=template.dtype)

    out = torch.empty((D, T), device=template.device, dtype=template.dtype)
    for t in range(T):
        out[:, t] = randn_like(template, generator=generator).reshape(-1)
    return out


def _apply_fir_2d(
    white: Tensor, kernel: Tensor, history: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Apply a causal FIR filter to [D, T] white noise."""
    D, _ = white.shape
    fir_order = kernel.numel()
    hist_len = fir_order - 1

    if history is None:
        history = torch.zeros((D, hist_len), device=white.device, dtype=white.dtype)
    else:
        if history.shape != (D, hist_len):
            raise ValueError(
                f"Expected history shape {(D, hist_len)}, got {history.shape}."
            )

    if hist_len > 0:
        seq = torch.cat([history, white], dim=1)
    else:
        seq = white

    # conv1d performs cross-correlation, so we flip to obtain causal FIR.
    weight = kernel.flip(0).view(1, 1, fir_order)
    out = F.conv1d(seq.unsqueeze(1), weight).squeeze(1)

    if hist_len > 0:
        new_history = seq[:, -hist_len:]
    else:
        new_history = seq[:, :0]
    return out, new_history


def pink_noise(
    *size: int,
    T: int,
    fir_order: int = 64,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    white_history: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_white_history: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Functional pink-noise generator using a causal FIR filter.

    Multi-step generation is vectorized: it samples all white inputs once and
    applies one convolution, instead of iterative per-step updates.
    """
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}.")
    if fir_order < 1:
        raise ValueError(f"fir_order must be >= 1, got {fir_order}.")

    hist_len = fir_order - 1
    if white_history is not None:
        if white_history.ndim < 1:
            raise ValueError("white_history must have at least one dimension.")
        if white_history.shape[-1] != hist_len:
            raise ValueError(
                f"white_history last dim must be {hist_len}, "
                f"got {white_history.shape[-1]}."
            )
        rest_shape = tuple(white_history.shape[:-1])
        sample_device = white_history.device
        sample_dtype = white_history.dtype
        history_flat = white_history.reshape(-1, hist_len)
        if len(size) != 0 and tuple(size) != rest_shape:
            raise ValueError(
                f"size={size} does not match white_history shape {rest_shape}."
            )
    else:
        if len(size) == 0:
            raise ValueError("Provide size or white_history for pink_noise.")
        rest_shape = tuple(size)
        sample_device = device or torch.device("cpu")
        sample_dtype = dtype or torch.get_default_dtype()
        history_flat = None

    if not torch.empty((), dtype=sample_dtype).is_floating_point():
        raise ValueError("pink_noise requires a floating dtype.")

    template = torch.empty(rest_shape, device=sample_device, dtype=sample_dtype)
    white = _white_noise_2d(template, T=T, generator=generator)
    kernel = _pink_fir_kernel(fir_order, device=sample_device, dtype=sample_dtype)
    out_flat, new_hist_flat = _apply_fir_2d(white, kernel, history_flat)
    out = _unflatten_td(out_flat, rest_shape)

    if not return_white_history:
        return out
    new_history = new_hist_flat.reshape(rest_shape + (hist_len,))
    return out, new_history


def pink_noise_like(
    like: Tensor,
    *,
    T: int,
    fir_order: int = 64,
    white_history: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_white_history: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Pink-noise convenience wrapper using a reference tensor."""
    dtype = like.dtype if like.is_floating_point() else torch.get_default_dtype()
    return pink_noise(
        *like.shape,
        T=T,
        fir_order=fir_order,
        device=like.device,
        dtype=dtype,
        white_history=white_history,
        generator=generator,
        return_white_history=return_white_history,
    )


class OUNoiseLayer(base.MemoryModule):
    """Ornstein-Uhlenbeck (OU) noise layer for temporally correlated noise.

    Exact discretization (sigma is stationary std):
        n_{t+1} = alpha * n_t + beta * eps_t
        alpha = exp(-dt/tau)
        beta  = sigma * sqrt(1 - exp(-2*dt/tau))
        eps_t ~ N(0, 1)

    Tensor conventions:
                - multi-step input x: [T, *batch_dims, *n_neuron]
                    (neuron dims must be trailing)
                - single-step input x: [*batch_dims, *n_neuron]
                - internal memory self.noise is aligned with x[0]
                    (state BEFORE current step/sequence)

        Multi-step backend:
                - convolution (scalar tau/sigma -> one conv1d; per-neuron
                    tau/sigma -> grouped conv1d, O(T^2*D))
        Determinism note:
                - multi-step uses a vectorized RNG path and convolution, so
                  exact equality with repeated single-step updates is not
                  guaranteed even when a generator is provided. Use single-step
                  loops if you need step-by-step equivalence; otherwise compare
                  statistical properties.

    Args:
        n_neuron: Number of neurons or shape of trailing neuron dims.
        sigma: scalar or tensor broadcastable to n_neuron.
        tau: scalar or tensor broadcastable to n_neuron.
        step_mode: 's' single-step, 'm' multi-step.
        trainable: whether sigma and tau are learnable.
        tau_min: clamp tau to this minimum (stability).
    """

    noise: Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        sigma: float | Tensor = 0.5,
        tau: float | Tensor = 10.0,
        step_mode: Literal["s", "m"] = "m",
        trainable: bool = False,
        *,
        stateful: bool = False,
        tau_min: float = 1e-6,
    ):
        super().__init__()
        if not stateful and step_mode == "s":
            raise ValueError("stateful must be True for single-step mode.")
        self.step_mode = step_mode
        self.stateful = stateful
        self.n_neuron, self.size = base.normalize_n_neuron(n_neuron)
        self.tau_min = float(tau_min)

        # Memory state: stored as "n_0" (state BEFORE current
        # step/sequence).
        # Will be expanded lazily to match x[0] if
        # register_memory does not include batch dims.
        if stateful:
            self.register_memory("noise", 0.0, self.n_neuron)

        sigma_t = torch.as_tensor(sigma)
        tau_t = torch.as_tensor(tau)

        if trainable:
            self.sigma = nn.Parameter(sigma_t)
            self.tau = nn.Parameter(tau_t)
        else:
            self.register_buffer("sigma", sigma_t)
            self.register_buffer("tau", tau_t)

    def single_step_forward(
        self, dt: float | None = None, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Single-step update of OU noise.

        Uses `self.noise` for device, dtype and shape. Returns the updated
        noise tensor (no input `x` is required or used).
        """
        assert self.stateful, "single_step_forward requires stateful=True"

        sigma, tau = self.sigma, self.tau
        dt: float = dt if dt is not None else environ.get("dt")

        alpha = torch.exp(-dt / tau)
        beta = sigma * torch.sqrt(1.0 - torch.exp(-2.0 * dt / tau))

        self.noise = alpha * self.noise + beta * randn_like(
            self.noise, generator=generator
        )
        return self.noise

    # ---------------- public multi-step ----------------

    def multi_step_forward(
        self,
        T: int,
        dt: float | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Public multi-step forward.

        Generates a noise sequence of length
        `T` and returns a tensor of shape [T, *noise_shape].
        """
        if T == 0:
            return torch.empty(
                (0,) + tuple(self.noise.shape),
                device=self.noise.device,
                dtype=self.noise.dtype,
            )

        if not hasattr(self, "noise"):
            raise RuntimeError(
                "OUNoiseLayer: noise buffer is not initialized. Call reset(...) first."
            )

        dt = dt if dt is not None else environ.get("dt")
        out = ou_noise(
            sigma=self.sigma,
            tau=self.tau,
            T=T,
            dt=dt,
            noise0=self.noise,
            generator=generator,
        )

        if self.stateful:
            # Update self.noise to the final state after the sequence.
            self.noise = out[-1]
        return out

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, sigma={self._format_repr_value(self.sigma)}, "
            f"tau={self._format_repr_value(self.tau)}, step_mode={self.step_mode}, "
            f"tau_min={self.tau_min}"
        )


class PoissonNoiseLayer(base.MemoryModule):
    """Poisson noise layer with per-step lambda = rate * dt.

    Supports single-step stateful sampling and vectorized multi-step
    sampling.
    """

    noise: Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        rate: float | Tensor = 1.0,
        step_mode: Literal["s", "m"] = "m",
        trainable: bool = False,
        *,
        stateful: bool = False,
    ):
        super().__init__()
        if not stateful and step_mode == "s":
            raise ValueError("stateful must be True for single-step mode.")
        self.step_mode = step_mode
        self.stateful = stateful
        self.n_neuron, self.size = base.normalize_n_neuron(n_neuron)

        if stateful:
            self.register_memory("noise", 0.0, self.n_neuron)

        rate_t = torch.as_tensor(rate)
        if trainable:
            self.rate = nn.Parameter(rate_t)
        else:
            self.register_buffer("rate", rate_t)

    def single_step_forward(
        self, dt: float | None = None, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Single-step Poisson sampling based on current dt."""
        assert self.stateful, "single_step_forward requires stateful=True"
        dt = dt if dt is not None else environ.get("dt")

        lam = torch.as_tensor(
            self.rate, device=self.noise.device, dtype=self.noise.dtype
        )
        lam = lam * float(dt)
        if torch.any(lam < 0):
            raise ValueError("Poisson rate * dt must be non-negative.")
        lam = torch.zeros_like(self.noise) + lam

        self.noise = torch.poisson(lam, generator=generator)
        return self.noise

    def multi_step_forward(
        self,
        T: int,
        dt: float | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Vectorized multi-step Poisson sampling."""
        if T == 0:
            return torch.empty(
                (0,) + tuple(self.noise.shape),
                device=self.noise.device,
                dtype=self.noise.dtype,
            )
        if not hasattr(self, "noise"):
            raise RuntimeError(
                "PoissonNoiseLayer: noise buffer is not initialized. "
                "Call init_state(...) first."
            )

        dt = dt if dt is not None else environ.get("dt")
        out = poisson_noise(
            *self.noise.shape,
            rate=self.rate,
            T=T,
            dt=dt,
            device=self.noise.device,
            dtype=self.noise.dtype,
            generator=generator,
        )
        if self.stateful:
            self.noise = out[-1]
        return out

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, rate={self._format_repr_value(self.rate)}, "
            f"step_mode={self.step_mode}"
        )


class PinkNoiseLayer(base.MemoryModule):
    """Pink (1/f) noise layer using a causal FIR filter.

    Single-step mode updates the FIR state one sample at a time. Multi-
    step mode uses one vectorized convolution pass over generated white
    noise.
    """

    noise: Tensor
    white_history: Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        fir_order: int = 64,
        step_mode: Literal["s", "m"] = "m",
        *,
        stateful: bool = False,
    ):
        super().__init__()
        if not stateful and step_mode == "s":
            raise ValueError("stateful must be True for single-step mode.")
        if fir_order < 1:
            raise ValueError(f"fir_order must be >= 1, got {fir_order}.")

        self.step_mode = step_mode
        self.stateful = stateful
        self.fir_order = int(fir_order)
        self.n_neuron, self.size = base.normalize_n_neuron(n_neuron)

        if stateful:
            self.register_memory("noise", 0.0, self.n_neuron)
            self.register_memory(
                "white_history", 0.0, self.n_neuron + (self.fir_order - 1,)
            )

    def single_step_forward(
        self, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Single-step pink-noise update using FIR history."""
        assert self.stateful, "single_step_forward requires stateful=True"
        out, new_hist = pink_noise(
            T=1,
            fir_order=self.fir_order,
            white_history=self.white_history,
            generator=generator,
            return_white_history=True,
        )
        self.noise = out[0]
        self.white_history = new_hist
        return self.noise

    def multi_step_forward(
        self,
        T: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Vectorized multi-step pink-noise generation."""
        if T == 0:
            return torch.empty(
                (0,) + tuple(self.noise.shape),
                device=self.noise.device,
                dtype=self.noise.dtype,
            )
        if not hasattr(self, "noise") or not hasattr(self, "white_history"):
            raise RuntimeError(
                "PinkNoiseLayer memories are not initialized. "
                "Call init_state(...) first."
            )

        out, new_hist = pink_noise(
            T=T,
            fir_order=self.fir_order,
            white_history=self.white_history,
            generator=generator,
            return_white_history=True,
        )
        if self.stateful:
            self.noise = out[-1]
            self.white_history = new_hist
        return out

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, fir_order={self.fir_order}, "
            f"step_mode={self.step_mode}"
        )
