"""GapJunction electrical synapse use cases and examples.

Gap junctions are electrical synapses that allow direct ion flow between neurons,
creating instantaneous, bidirectional coupling. Common use cases include:

1. Synchronizing neuronal populations (e.g., interneuron networks)
2. Modeling specific circuits (retinal ganglion cells, thalamic relay neurons)
3. Rapid signal propagation without synaptic delay
4. Bidirectional coupling between neuron pairs
"""

import platform

import matplotlib.pyplot as plt
import pytest
import torch

from btorch.models.synapse import GapJunction
from btorch.utils.file import save_fig


def _identity_linear(n: int) -> torch.nn.Linear:
    """Create a linear layer with identity weights."""
    linear = torch.nn.Linear(n, n, bias=False)
    torch.nn.init.eye_(linear.weight)
    return linear


def test_gap_junction_basic_coupling():
    """Basic use case: two neurons with electrical coupling.

    When neuron A is at higher potential than neuron B, current flows from
    A to B, pulling their voltages toward each other. This is the fundamental
    mechanism for electrical coupling in networks.
    """
    # Two neurons with symmetric coupling (identity weights for predictable output)
    gap = GapJunction(n_neuron=2, g_gap=0.5, linear=_identity_linear(2))

    # Neuron 0 at 10mV, neuron 1 at 0mV
    v_pre = torch.tensor([[10.0, 0.0]])
    v_post = torch.tensor([[0.0, 0.0]])

    i_gap = gap(v_pre, v_post)

    # Current flows out of neuron 0 (negative), into neuron 1 would be from
    # another gap junction with reversed pre/post
    # Here: delta_v = [0-10, 0-0] = [-10, 0]
    # I_gap = 0.5 * [-10, 0] = [-5, 0]
    expected = torch.tensor([[-5.0, 0.0]])
    torch.testing.assert_close(i_gap, expected, atol=1e-6, rtol=0.0)

    # Visualize the coupling: voltage difference drives current
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Voltage state
    ax = axes[0]
    neurons = ["Neuron 0", "Neuron 1"]
    v_vals = v_pre[0].tolist()
    colors = ["#d62728", "#2ca02c"]  # red for high, green for low
    ax.bar(neurons, v_vals, color=colors, edgecolor="black")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Pre-synaptic Voltage State")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Current flow
    ax = axes[1]
    i_vals = i_gap[0].tolist()
    colors_i = ["#d62728" if i < 0 else "#2ca02c" for i in i_vals]
    ax.bar(neurons, i_vals, color=colors_i, edgecolor="black")
    ax.set_ylabel("Gap Junction Current (pA)")
    ax.set_title("Current Flow (negative = losing charge)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, name="gap_junction_basic_coupling")
    plt.close(fig)


def test_gap_junction_bidirectional_symmetry():
    """Gap junctions are bidirectional: I(A->B) = -I(B->A).

    This property ensures energy conservation and reciprocal coupling.
    In a network, each gap junction should be applied twice (once in each
    direction) or use symmetric weight matrices.
    """
    gap = GapJunction(n_neuron=2, g_gap=1.0)

    v_a = torch.tensor([[10.0, 0.0]])
    v_b = torch.tensor([[0.0, 10.0]])

    # Current when A is "pre" and B is "post"
    i_a_to_b = gap(v_a, v_b)
    # Current when B is "pre" and A is "post"
    i_b_to_a = gap(v_b, v_a)

    # Should be equal and opposite
    torch.testing.assert_close(i_a_to_b, -i_b_to_a, atol=1e-6, rtol=0.0)


def test_gap_junction_synchronization():
    """Use case: gap junctions synchronize coupled neurons over time.

    Demonstrates how gap junctions pull coupled neurons toward each other.
    Neuron with higher voltage loses charge (negative current) to neuron
    with lower voltage (positive current when viewed from the other side).
    """
    gap = GapJunction(n_neuron=2, g_gap=0.5)

    # Scenario: neuron 0 at 10mV, neuron 1 at 0mV
    v_pre = torch.tensor([[10.0, 0.0]])
    v_post = torch.tensor([[0.0, 0.0]])

    # Current from neuron 0's perspective (receiving from network at v_post)
    # delta_v = [0-10, 0-0] = [-10, 0]
    # I_gap = 0.5 * [-10, 0] = [-5, 0]
    i_gap = gap(v_pre, v_post)

    # Neuron 0 receives negative current (loses charge to the network)
    assert (
        i_gap[0, 0] < 0
    ), f"Higher voltage neuron should lose current, got {i_gap[0, 0].item()}"

    # Reverse: if we swap pre/post, neuron 1 now receives from neuron 0
    i_gap_reverse = gap(v_post, v_pre)
    # delta_v = [10-0, 0-0] = [10, 0]
    # I_gap = 0.5 * [10, 0] = [5, 0]
    # Neuron 0 receives positive current (gains charge from network)
    assert (
        i_gap_reverse[0, 0] > 0
    ), f"Lower voltage neuron should gain current, got {i_gap_reverse[0, 0].item()}"

    # The currents are equal and opposite, demonstrating synchronization
    torch.testing.assert_close(i_gap[0, 0], -i_gap_reverse[0, 0], atol=1e-6, rtol=0.0)

    # Visualize bidirectional current flow
    fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = ["Neuron 0\n(higher V)", "Neuron 1\n(lower V)"]
    current_0_to_1 = [i_gap[0, 0].item(), i_gap[0, 1].item()]
    current_1_to_0 = [i_gap_reverse[0, 0].item(), i_gap_reverse[0, 1].item()]

    x = range(len(scenarios))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        current_0_to_1,
        width,
        label="View from V=[0,0]",
        color="#1f77b4",
    )
    ax.bar(
        [i + width / 2 for i in x],
        current_1_to_0,
        width,
        label="View from V=[10,0]",
        color="#ff7f0e",
    )

    ax.set_ylabel("Gap Junction Current (pA)")
    ax.set_title("Bidirectional Current Flow Demonstrates Symmetry")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, name="gap_junction_bidirectional_symmetry")
    plt.close(fig)


def test_gap_junction_ring_network():
    """Use case: ring network with nearest-neighbor coupling.

    Common in modeling electrically coupled interneuron networks where each
    neuron connects to its neighbors (e.g., cortical fast-spiking interneurons).
    """
    n = 4
    # Create coupling matrix: each neuron connects to neighbors
    # Weight matrix with off-diagonal coupling
    linear = torch.nn.Linear(n, n, bias=False)
    with torch.no_grad():
        # Nearest-neighbor coupling in a ring
        w = torch.zeros(n, n)
        for i in range(n):
            w[i, (i - 1) % n] = 0.5  # left neighbor
            w[i, (i + 1) % n] = 0.5  # right neighbor
        linear.weight.copy_(w)

    gap = GapJunction(n_neuron=n, g_gap=1.0, linear=linear)

    # One neuron at high voltage, others at rest (simulating neuron 0 spiking)
    v_pre = torch.zeros(1, n)
    v_pre[0, 0] = 10.0
    v_post = torch.zeros(1, n)  # Other neurons at rest

    # Compute gap current: I = g * W * (v_post - v_pre)
    # With our setup: delta_v = [0-10, 0-0, 0-0, 0-0] = [-10, 0, 0, 0]
    # W @ delta_v gives current to each neuron
    i_gap = gap(v_pre, v_post)

    # With delta_v = v_post - v_pre = [0-10, 0-0, 0-0, 0-0] = [-10, 0, 0, 0]
    # W[i,:] @ delta_v gives current to neuron i
    # i_0 = 0 (no neighbors with non-zero voltage difference)
    # i_1 = 0.5 * (-10) = -5 (receives from neuron 0)
    # i_3 = 0.5 * (-10) = -5 (receives from neuron 0)
    # Verify neighbors receive equal current from neuron 0
    assert (
        abs(i_gap[0, 1] - i_gap[0, 3]) < 1e-6
    ), "Neighbors should receive equal current"
    assert (
        abs(i_gap[0, 1]) > 0
    ), f"Neighbor 1 should have current, got {i_gap[0, 1].item()}"
    assert (
        abs(i_gap[0, 3]) > 0
    ), f"Neighbor 3 should have current, got {i_gap[0, 3].item()}"
    # Distant neuron (2) should have no direct coupling
    assert (
        abs(i_gap[0, 2]) < 1e-6
    ), f"Distant neuron should have no current, got {i_gap[0, 2].item()}"

    # Visualize ring network current distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Voltage input
    ax = axes[0]
    neuron_labels = [f"N{i}" for i in range(n)]
    v_vals = v_pre[0].tolist()
    colors_v = ["#d62728" if v > 0 else "#cccccc" for v in v_vals]
    bars = ax.bar(neuron_labels, v_vals, color=colors_v, edgecolor="black")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Ring Network: Neuron 0 Active (10mV)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Add ring connection visualization
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(
            f"N{i}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Current distribution
    ax = axes[1]
    i_vals = i_gap[0].tolist()
    colors_i = [
        "#d62728" if i < 0 else "#2ca02c" if i > 0 else "#cccccc" for i in i_vals
    ]
    ax.bar(neuron_labels, i_vals, color=colors_i, edgecolor="black")
    ax.set_ylabel("Gap Junction Current (pA)")
    ax.set_title("Current Distribution (neighbors N1,N3 receive from N0)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, name="gap_junction_ring_network")
    plt.close(fig)


def test_gap_junction_learnable_conductance():
    """Use case: training gap junction conductances as network parameters.

    In some models, gap junction strengths are learned during training to
    optimize network synchronization or information flow.
    """
    gap = GapJunction(n_neuron=3, g_gap=0.5)

    # Make g_gap trainable
    gap.g_gap = torch.nn.Parameter(gap.g_gap)

    v_pre = torch.tensor([[10.0, 5.0, 0.0]], requires_grad=True)
    v_post = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

    i_gap = gap(v_pre, v_post)
    loss = (i_gap**2).sum()
    loss.backward()

    # Gradient should flow to g_gap
    assert gap.g_gap.grad is not None
    # Gradient should push toward reducing current magnitude
    assert gap.g_gap.grad.item() != 0


def test_gap_junction_multi_neuron_batch():
    """Use case: processing multiple neurons across batch dimension.

    In batched simulations, we often compute gap junction currents for
    multiple network instances simultaneously (e.g., different trials
    or parameter sets).
    """
    # Use identity weights for predictable output verification
    gap = GapJunction(n_neuron=4, g_gap=0.2, linear=_identity_linear(4))

    batch_size = 8
    v_pre = torch.randn(batch_size, 4)
    v_post = torch.randn(batch_size, 4)

    i_gap = gap(v_pre, v_post)

    assert i_gap.shape == (batch_size, 4)

    # Each batch element computed independently
    for b in range(batch_size):
        expected = 0.2 * (v_post[b] - v_pre[b])
        torch.testing.assert_close(i_gap[b], expected, atol=1e-6, rtol=0.0)


def test_gap_junction_time_series():
    """Use case: gap junction currents in time-series simulations.

    In spiking neural network simulations, gap junction currents are
    computed at each timestep based on instantaneous voltage differences.
    Uses identity weights for predictable current calculation.
    """
    # Use identity weights: I_gap = g_gap * (v_post - v_pre)
    gap = GapJunction(n_neuron=2, g_gap=0.5, linear=_identity_linear(2))

    T = 100  # timesteps
    dt = 0.1  # ms per timestep
    # Simulate oscillating voltages
    t = torch.arange(T, dtype=torch.float32)
    v_pre_seq = torch.stack(
        [
            torch.sin(t * 0.1),  # Neuron 0 oscillates
            torch.cos(t * 0.1),  # Neuron 1 oscillates with phase shift
        ],
        dim=1,
    ).unsqueeze(1)  # (T, 1, 2)
    v_post_seq = torch.zeros_like(v_pre_seq)

    i_gap_seq = gap.multi_step_forward(v_pre_seq, v_post_seq)

    assert i_gap_seq.shape == (T, 1, 2)

    # With identity weights and g_gap=0.5: I_gap = 0.5 * (0 - sin(t)) = -0.5 * sin(t)
    # Peak current magnitude should be 0.5 (when sin(t) = 1 or -1)
    # Verify at peaks of oscillation, current is maximal
    # Peak of sine is at t=~16 (pi/2 / 0.1)
    peak_idx = int(3.14159 / 2 / 0.1)
    expected_peak_current = 0.5  # g_gap * 1.0 (max voltage difference)
    actual = i_gap_seq[peak_idx, 0, 0].item()
    assert (
        abs(i_gap_seq[peak_idx, 0, 0]) > 0.4
    ), f"Current should be ~{expected_peak_current} at peak, got {actual}"

    # Visualize time series: voltage and current over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    time_ms = (t * dt).numpy()

    # Voltage traces
    ax = axes[0]
    ax.plot(
        time_ms,
        v_pre_seq[:, 0, 0].detach().numpy(),
        label="Neuron 0 (sin)",
        color="#1f77b4",
    )
    ax.plot(
        time_ms,
        v_pre_seq[:, 0, 1].detach().numpy(),
        label="Neuron 1 (cos)",
        color="#ff7f0e",
    )
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Oscillating Neuron Voltages")
    ax.legend()
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Gap junction current
    ax = axes[1]
    ax.plot(
        time_ms,
        i_gap_seq[:, 0, 0].detach().numpy(),
        label="Neuron 0 current",
        color="#1f77b4",
    )
    ax.plot(
        time_ms,
        i_gap_seq[:, 0, 1].detach().numpy(),
        label="Neuron 1 current",
        color="#ff7f0e",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Gap Junction Current (pA)")
    ax.set_title("Gap Junction Current (I = g * (V_post - V_pre))")
    ax.legend()
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, name="gap_junction_time_series")
    plt.close(fig)


def test_gap_junction_zero_conductance_no_effect():
    """Edge case: zero conductance should produce no coupling.

    Useful for ablation studies or conditionally disabling gap junctions.
    """
    gap = GapJunction(n_neuron=3, g_gap=0.0)

    v_pre = torch.randn(2, 3)
    v_post = torch.randn(2, 3)

    i_gap = gap(v_pre, v_post)

    assert torch.allclose(i_gap, torch.zeros_like(i_gap))


def test_gap_junction_equal_voltage_no_current():
    """Edge case: equal voltages produce no current (Ohm's law).

    When coupled neurons have the same membrane potential, there is no
    driving force for ion flow, regardless of conductance strength.
    """
    gap = GapJunction(n_neuron=4, g_gap=1.0)

    v = torch.randn(2, 4)
    i_gap = gap(v, v.clone())

    assert torch.allclose(i_gap, torch.zeros_like(i_gap), atol=1e-6)


def test_gap_junction_2d_spatial_layout():
    """Use case: 2D spatial arrangement of neurons (e.g., retinal mosaic).

    Retinal neurons are often arranged in 2D mosaics with gap junction
    coupling between nearby cells. This tests spatial layout handling.
    """
    gap = GapJunction(n_neuron=(4, 4), g_gap=0.1)

    # 2D voltage map with a hotspot
    v_pre = torch.zeros(1, 4, 4)
    v_pre[0, 1:3, 1:3] = 10.0  # 2x2 hotspot
    v_post = torch.zeros(1, 4, 4)

    i_gap = gap(v_pre, v_post)

    # Current should flow out of the hotspot
    assert i_gap.shape == (1, 4, 4)
    assert torch.all(i_gap[0, 1:3, 1:3] < 0), "Hotspot should lose current"

    # Visualize 2D spatial layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Voltage heatmap
    ax = axes[0]
    im = ax.imshow(v_pre[0].detach().numpy(), cmap="RdYlGn_r", vmin=-2, vmax=12)
    ax.set_title("2D Voltage Map (Hotspot Center)")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{v_pre[0, i, j].item():.1f}",
                ha="center",
                va="center",
                color="white" if v_pre[0, i, j] > 5 else "black",
            )
    plt.colorbar(im, ax=ax, label="Voltage (mV)")

    # Current heatmap
    ax = axes[1]
    im = ax.imshow(i_gap[0].detach().numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Gap Junction Current Map")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{i_gap[0, i, j].item():.2f}",
                ha="center",
                va="center",
                color="white" if abs(i_gap[0, i, j]) > 0.5 else "black",
            )
    plt.colorbar(im, ax=ax, label="Current (pA)")

    fig.tight_layout()
    save_fig(fig, name="gap_junction_2d_spatial")
    plt.close(fig)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_gap_junction_compile_compatibility():
    """Test that GapJunction works with torch.compile for performance.

    Important for large-scale simulations where compilation can provide
    significant speedups.
    """
    from tests.utils.compile import compile_or_skip

    gap = GapJunction(n_neuron=4, g_gap=0.3)
    gap_compiled = compile_or_skip(gap)

    v_pre = torch.randn(2, 4)
    v_post = torch.randn(2, 4)

    i_eager = gap(v_pre, v_post)
    i_compiled = gap_compiled(v_pre, v_post)

    torch.testing.assert_close(i_eager, i_compiled, atol=1e-6, rtol=0.0)
