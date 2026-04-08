import pytest
import torch

from btorch.datasets.noise import (
    OUNoiseLayer,
    PinkNoiseLayer,
    PoissonNoiseLayer,
    ou_noise,
    ou_noise_like,
    pink_noise,
    pink_noise_like,
    poisson_noise,
    poisson_noise_like,
)
from btorch.models import environ


@pytest.mark.parametrize(
    ("layer_factory", "param_names"),
    [
        pytest.param(
            lambda: OUNoiseLayer(
                n_neuron=5,
                sigma=0.5,
                tau=10.0,
                trainable_param={"sigma", "tau"},
                stateful=True,
                step_mode="s",
            ),
            ("sigma", "tau"),
            id="ou",
        ),
        pytest.param(
            lambda: PoissonNoiseLayer(
                n_neuron=5,
                rate=2.0,
                trainable_param={"rate"},
                stateful=True,
                step_mode="s",
            ),
            ("rate",),
            id="poisson",
        ),
    ],
)
def test_noise_layer_trainable(layer_factory, param_names):
    with environ.context(dt=1.0):
        noise_layer = layer_factory()
        noise_layer.init_state(batch_size=10)
        for name in param_names:
            assert isinstance(getattr(noise_layer, name), torch.nn.Parameter)

        # Check gradients
        # single-step API no longer requires an input tensor
        y = noise_layer.single_step_forward()
        y.sum().backward()

        for name in param_names:
            grad = getattr(noise_layer, name).grad
            assert grad is not None
            assert torch.isfinite(grad).all()
        assert noise_layer.noise.shape == (10, 5)


def test_ou_noise_functional_like():
    """Functional APIs should follow OU statistical properties."""
    dt = 1.0
    like = torch.zeros(2, 3)
    sigma = torch.tensor(0.5)
    tau = torch.tensor(10.0)

    # Like-style wrapper defaults to randn_like for noise0.
    gen = torch.Generator().manual_seed(123)
    y_like = ou_noise_like(like, sigma, tau, T=3, dt=dt, generator=gen)
    assert y_like.shape == (3, 2, 3)

    # Statistical sanity: long-run mean ~ 0, std ~ sigma for stationary OU.
    gen = torch.Generator().manual_seed(321)
    y_long = ou_noise(
        *like.shape,
        sigma=sigma,
        tau=tau,
        T=2000,
        dt=dt,
        generator=gen,
    )
    flat = y_long.reshape(-1)
    mean = flat.mean().item()
    std = flat.std(unbiased=True).item()
    assert abs(mean) < 0.05
    assert abs(std - float(sigma)) < 0.05

    # Autocorrelation sanity: consecutive steps should be positively correlated.
    y_pair = y_long[1:].reshape(-1)
    y_prev = y_long[:-1].reshape(-1)
    y_pair = y_pair - y_pair.mean()
    y_prev = y_prev - y_prev.mean()
    corr = (y_pair * y_prev).mean() / (y_pair.std() * y_prev.std())
    assert corr > 0.1


def test_poisson_noise_functional_like():
    """Poisson functional APIs should match basic count statistics."""
    like = torch.zeros(2, 3)
    rate = 2.5
    dt = 1.0

    gen = torch.Generator().manual_seed(11)
    y_like = poisson_noise_like(like, rate=rate, T=5, dt=dt, generator=gen)
    assert y_like.shape == (5, 2, 3)

    gen = torch.Generator().manual_seed(12)
    y_long = poisson_noise(
        *like.shape,
        rate=rate,
        T=3000,
        dt=dt,
        generator=gen,
    )
    flat = y_long.reshape(-1)

    # Poisson outputs non-negative integer counts (stored in floating dtype).
    assert torch.all(flat >= 0)
    assert torch.allclose(flat, flat.round(), atol=1e-6)

    # For Poisson(lambda), mean ~= variance ~= lambda.
    mean = flat.mean().item()
    var = flat.var(unbiased=True).item()
    assert abs(mean - rate * dt) < 0.12
    assert abs(var - rate * dt) < 0.2


def test_poisson_noise_layer_multistep():
    """Poisson layer should support vectorized multi-step and state updates."""
    with environ.context(dt=1.0):
        layer = PoissonNoiseLayer(n_neuron=4, rate=3.0, step_mode="m", stateful=True)
        layer.init_state(batch_size=2)

        gen = torch.Generator().manual_seed(13)
        y = layer.multi_step_forward(7, generator=gen)
        assert y.shape == (7, 2, 4)
        assert torch.all(y >= 0)
        assert torch.allclose(layer.noise, y[-1])


def test_pink_noise_functional_and_layer():
    """Pink noise should follow 1/f trend and support FIR single-step state."""
    like = torch.zeros(2, 3)

    gen = torch.Generator().manual_seed(21)
    y_like = pink_noise_like(like, T=6, fir_order=32, generator=gen)
    assert y_like.shape == (6, 2, 3)

    gen = torch.Generator().manual_seed(22)
    y_long = pink_noise(*like.shape, T=4096, fir_order=64, generator=gen)
    assert y_long.shape == (4096, 2, 3)

    # 1/f behavior: log PSD slope should be close to -1 over a mid-frequency band.
    channels = y_long.reshape(y_long.shape[0], -1).transpose(0, 1)
    power = torch.fft.rfft(channels, dim=-1).abs().pow(2).mean(dim=0)
    freqs = torch.fft.rfftfreq(y_long.shape[0], d=1.0)
    lo = max(1, int(0.01 * freqs.numel()))
    hi = int(0.4 * freqs.numel())
    fit_f = freqs[lo:hi]
    fit_p = power[lo:hi].clamp_min(1e-20)
    design = torch.stack([torch.log(fit_f), torch.ones_like(fit_f)], dim=1)
    slope = torch.linalg.lstsq(design, torch.log(fit_p).unsqueeze(1)).solution[0, 0]
    assert -1.2 < slope.item() < -0.8

    # Single-step FIR mode keeps state and yields valid output shape.
    layer = PinkNoiseLayer(n_neuron=3, fir_order=16, step_mode="s", stateful=True)
    layer.init_state(batch_size=2)
    gen = torch.Generator().manual_seed(23)
    y_s = [layer.single_step_forward(generator=gen) for _ in range(8)]
    y_s = torch.stack(y_s)
    assert y_s.shape == (8, 2, 3)
    assert torch.allclose(layer.noise, y_s[-1])
