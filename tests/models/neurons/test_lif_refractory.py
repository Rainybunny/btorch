import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.lif import LIF


def _run_two_steps(neuron: LIF, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Initialize state so the first step starts from v_reset.
    init_net_state(neuron)
    with environ.context(dt=1.0):
        spike_0 = neuron(x)
        spike_1 = neuron(x)
    return spike_0, spike_1


def test_lif_refractory_gates_spikes():
    # Use a strong constant input so the neuron spikes on the first step.
    x = torch.tensor([10.0])

    neuron = LIF(
        n_neuron=1,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        tau=1.0,
        tau_ref=2.0,
    )
    spike_0, spike_1 = _run_two_steps(neuron, x)
    assert spike_0.item() > 0.0
    # tau_ref > dt keeps the refractory counter positive for the next step.
    assert spike_1.item() == 0.0

    # With tau_ref=None, the refractory buffer is not registered and spikes
    # are not gated on subsequent steps.
    neuron_no_ref = LIF(
        n_neuron=1,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        tau=1.0,
        tau_ref=None,
    )
    spike_0, spike_1 = _run_two_steps(neuron_no_ref, x)
    assert "refractory" not in dict(neuron_no_ref.named_buffers())
    assert spike_1.item() > 0.0
