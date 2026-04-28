"""Consistency tests for ChemSynModel0Gate.

Run with:
    conda run -n spike_gpu python -m pytest btorch/models/test_chemsyn_model0_gate.py -v
or:
    python btorch/models/test_chemsyn_model0_gate.py
"""

import math
import sys
from pathlib import Path

import torch
# Insert repo root so the local btorch package takes precedence.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from btorch.models.synapse import ChemSynModel0Gate
from btorch.models import environ


# ---------------------------------------------------------------------------
# Reference implementation (manual loop, mirrors demo script logic)
# ---------------------------------------------------------------------------

def _manual_chemsyn(
    spikes: torch.Tensor,        # (T, n_neuron) binary int32
    steps_trans: torch.Tensor,   # (n_neuron,) int32
    tau_syn: float,
    dt: float,
) -> torch.Tensor:
    """Reference ChemSyn model-0 loop; returns (T, n_neuron) release tensor."""
    n_neuron = spikes.shape[1]
    k_trans = 1.0 / steps_trans.float().clamp(min=1)
    s_pre_decay = math.exp(-dt / max(tau_syn, 1e-9))

    s_pre = torch.zeros(n_neuron)
    trans_left = torch.zeros(n_neuron, dtype=torch.int32)
    releases = []

    for t in range(spikes.shape[0]):
        fired = spikes[t].to(torch.int32)
        trans_left = trans_left + fired * steps_trans
        active = (trans_left > 0).float()
        release = active * k_trans * (1.0 - s_pre)
        s_pre = torch.where(active.bool(),
                            s_pre + k_trans * (1.0 - s_pre),
                            s_pre) * s_pre_decay
        trans_left = (trans_left - active.to(torch.int32)).clamp(min=0)
        releases.append(release.clone())

    return torch.stack(releases)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

DT = 0.1        # ms
TAU = 3.0       # ms
N = 8
STEPS = 5       # uniform steps_trans for most tests


def _make_gate(steps_val: int = STEPS, n: int = N) -> ChemSynModel0Gate:
    steps = torch.full((n,), steps_val, dtype=torch.int32)
    gate = ChemSynModel0Gate(n_neuron=n, steps_trans=steps, tau_syn=TAU)
    gate.init_state()
    return gate


def test_numerical_equivalence_random():
    """Gate output must match manual reference loop for random spike train."""
    torch.manual_seed(0)
    T = 50
    spikes = (torch.rand(T, N) < 0.3).to(torch.int32)

    # Reference
    steps = torch.full((N,), STEPS, dtype=torch.int32)
    ref = _manual_chemsyn(spikes, steps, TAU, DT)

    # Module
    gate = _make_gate()
    out_list = []
    with environ.context(dt=DT):
        for t in range(T):
            r = gate(spikes[t].float())
            out_list.append(r.detach().clone())
    mod_out = torch.stack(out_list)

    assert torch.allclose(ref.float(), mod_out.float(), atol=1e-6), (
        f"Max abs diff: {(ref - mod_out).abs().max()}"
    )


def test_saturation_at_high_firing_rate():
    """At very high firing rate, s_pre should approach 1 and release approach 0."""
    gate = _make_gate(steps_val=STEPS)
    # Fire every step for 200 steps
    z_on = torch.ones(N)
    with environ.context(dt=DT):
        for _ in range(200):
            gate(z_on)
    # s_pre should be near 1
    assert float(gate.s_pre.mean()) > 0.7, f"s_pre mean={float(gate.s_pre.mean()):.4f}"


def test_release_zero_when_silent():
    """After firing, once silent for enough steps, release should be 0."""
    gate = _make_gate(steps_val=STEPS)
    z_on = torch.ones(N)
    z_off = torch.zeros(N)
    with environ.context(dt=DT):
        gate(z_on)  # fire once
        for _ in range(STEPS + 2):
            r = gate(z_off)
    # After release window expires, release must be exactly 0
    assert float(r.sum()) == 0.0, f"Expected 0 release, got {float(r.sum())}"


def test_from_ei_populations_equivalence():
    """from_ei_populations should match manually constructed steps_trans."""
    n_e, n_i = 6, 4
    Dt_ampa, Dt_gaba = 1.0, 0.5  # ms
    steps_e = max(1, round(Dt_ampa / DT))   # 10
    steps_i = max(1, round(Dt_gaba / DT))   # 5

    gate_manual = ChemSynModel0Gate(
        n_neuron=n_e + n_i,
        steps_trans=torch.cat([
            torch.full((n_e,), steps_e, dtype=torch.int32),
            torch.full((n_i,), steps_i, dtype=torch.int32),
        ]),
        tau_syn=TAU,
    )
    gate_factory = ChemSynModel0Gate.from_ei_populations(
        n_e=n_e, n_i=n_i,
        Dt_trans_ampa=Dt_ampa, Dt_trans_gaba=Dt_gaba,
        tau_syn=TAU, dt=DT,
    )

    assert torch.equal(gate_manual.steps_trans, gate_factory.steps_trans), (
        f"steps_trans mismatch: {gate_manual.steps_trans} vs {gate_factory.steps_trans}"
    )
    assert torch.allclose(gate_manual.k_trans, gate_factory.k_trans), (
        "k_trans mismatch"
    )


def test_ei_separation_different_release_duration():
    """E and I neurons in a from_ei network must have different release windows."""
    n_e, n_i = 4, 4
    gate = ChemSynModel0Gate.from_ei_populations(
        n_e=n_e, n_i=n_i,
        Dt_trans_ampa=2.0, Dt_trans_gaba=0.5,
        tau_syn=TAU, dt=DT,
    )
    gate.init_state()
    steps = gate.steps_trans
    assert int(steps[:n_e].unique().item()) != int(steps[n_e:].unique().item()), (
        "E and I neurons must have different steps_trans"
    )
    # Fire once, then count how many steps each half stays active
    z = torch.ones(n_e + n_i)
    active_counts = torch.zeros(n_e + n_i, dtype=torch.int32)
    with environ.context(dt=DT):
        gate(z)  # fire
        for _ in range(30):
            r = gate(torch.zeros(n_e + n_i))
            active_counts += (r > 0).to(torch.int32)
    # E neurons (steps=20) should stay active longer than I neurons (steps=5)
    assert active_counts[:n_e].float().mean() > active_counts[n_e:].float().mean(), (
        "E neurons should release longer than I neurons"
    )


def test_reset_clears_state():
    """After reset(), s_pre and trans_left should return to zero."""
    gate = _make_gate()
    z = torch.ones(N)
    with environ.context(dt=DT):
        for _ in range(10):
            gate(z)
    gate.reset()
    assert float(gate.s_pre.sum()) == 0.0
    assert int(gate.trans_left.sum()) == 0


def test_extra_repr():
    gate = _make_gate()
    r = repr(gate)
    assert "ChemSynModel0Gate" in r


if __name__ == "__main__":
    tests = [
        test_numerical_equivalence_random,
        test_saturation_at_high_firing_rate,
        test_release_zero_when_silent,
        test_from_ei_populations_equivalence,
        test_ei_separation_different_release_duration,
        test_reset_clears_state,
        test_extra_repr,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            raise
    print("\nAll tests passed.")
