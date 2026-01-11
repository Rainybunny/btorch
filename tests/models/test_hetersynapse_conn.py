"""Tests for hetersynapse connection and constraint functionality.

This test module covers:
1. make_hetersynapse_conn with dict returns and NaN handling
2. make_hetersynapse_constraint with different constraint modes
3. make_hetersynapse_constrained_conn integration
4. HeterSynapsePSC.get_psc with autodetection
5. SparseConstrainedConn enhancements (from_hetersynapse, helper methods)

Tests also serve as examples and documentation for how to use these features.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import torch

from btorch.connectome.connection import (
    make_hetersynapse_conn,
    make_hetersynapse_constrained_conn,
    make_hetersynapse_constraint,
)
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.linear import SparseConstrainedConn
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC
from tests.utils.file import save_fig


def create_test_neurons(n_neurons=100, seed=42):
    """Create test neuron DataFrame with cell types and receptor types.

    This helper creates realistic test data for synapse testing.

    Args:
        n_neurons: Number of neurons to create
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: root_id, simple_id, cell_type, EI
    """
    np.random.seed(seed)

    # Create neuron IDs
    neurons = pd.DataFrame(
        {
            "root_id": np.arange(1000, 1000 + n_neurons),
            "simple_id": np.arange(n_neurons),
        }
    )

    # Assign cell types (e.g., different neuron classes)
    n_cell_types = 5
    neurons["cell_type"] = [f"type_{i % n_cell_types}" for i in range(n_neurons)]

    # Assign receptor types (E or I)
    neurons["EI"] = np.random.choice(["E", "I"], size=n_neurons, p=[0.7, 0.3])

    return neurons


def create_test_connections(neurons, density=0.1, seed=42):
    """Create test connection DataFrame.

    Args:
        neurons: DataFrame from create_test_neurons
        density: Connection density (fraction of possible connections)
        seed: Random seed

    Returns:
        DataFrame with columns: pre_root_id, post_root_id, pre_simple_id,
        post_simple_id, syn_count
    """
    np.random.seed(seed)

    n_neurons = len(neurons)
    n_connections = int(n_neurons**2 * density)

    # Create random connections
    pre_idx = np.random.choice(n_neurons, size=n_connections)
    post_idx = np.random.choice(n_neurons, size=n_connections)

    # Remove self-connections
    mask = pre_idx != post_idx
    pre_idx = pre_idx[mask]
    post_idx = post_idx[mask]

    connections = pd.DataFrame(
        {
            "pre_simple_id": pre_idx,
            "post_simple_id": post_idx,
            "syn_count": np.random.randint(1, 10, size=len(pre_idx)),
        }
    )

    # Add root IDs
    connections["pre_root_id"] = connections["pre_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )
    connections["post_root_id"] = connections["post_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )

    return connections


def test_make_hetersynapse_conn_dict_return_neuron_mode():
    """Test make_hetersynapse_conn with return_dict=True in neuron mode.

    This demonstrates how to get separate matrices for each receptor
    type pair, allowing manual scaling or manipulation before combining.
    """
    neurons = create_test_neurons(n_neurons=50)
    connections = create_test_connections(neurons, density=0.15)

    # Get dict return
    conn_dict, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=True,
    )

    # Verify it's a dict
    assert isinstance(conn_dict, dict)

    # Check structure
    assert len(conn_dict) > 0

    # Each key should be (pre_receptor, post_receptor) tuple
    for key, mat in conn_dict.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert all(isinstance(k, str) for k in key)

        # Each value should be a sparse matrix
        assert scipy.sparse.issparse(mat)
        assert mat.shape == (len(neurons), len(neurons))

    # receptor_idx should have pre/post receptor types
    assert "pre_receptor_type" in receptor_idx.columns
    assert "post_receptor_type" in receptor_idx.columns
    assert "receptor_index" in receptor_idx.columns

    print(f"✓ Dict return created {len(conn_dict)} receptor type pairs")
    print(f"  Receptor pairs: {list(conn_dict.keys())}")


def test_make_hetersynapse_conn_nan_handling():
    """Test NaN handling in make_hetersynapse_conn.

    This demonstrates the dropna parameter and error handling for
    missing receptor type data.
    """
    neurons = create_test_neurons(n_neurons=30)
    connections = create_test_connections(neurons, density=0.2)

    # Add some NaN receptor types
    neurons.loc[5:8, "EI"] = np.nan

    # Should raise error by default (dropna='error')
    with pytest.raises(ValueError, match="NaN receptor types found"):
        make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            dropna="error",
        )

    # Test dropna='filter' - removes connections involving NaN neurons
    with pytest.warns(UserWarning, match="Filtered out .* connections involving"):
        conn_mat_filter, receptor_idx_filter = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            dropna="filter",
        )

    assert scipy.sparse.issparse(conn_mat_filter)
    # Neuron count should be preserved
    assert conn_mat_filter.shape[0] == len(neurons)
    print("✓ dropna='filter' works correctly (preserves neuron count)")

    # Test dropna='unknown' - treats NaN as a separate receptor type
    with pytest.warns(UserWarning, match="Treating .* neurons with NaN receptor types"):
        conn_mat_unknown, receptor_idx_unknown = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            dropna="unknown",
        )

    assert scipy.sparse.issparse(conn_mat_unknown)
    # Neuron count should be preserved
    assert conn_mat_unknown.shape[0] == len(neurons)
    # Should have more receptor type pairs (including NaN combinations)
    assert len(receptor_idx_unknown) > len(receptor_idx_filter)
    print("✓ dropna='unknown' works correctly (NaN as receptor type)")

    print("✓ NaN handling works correctly")


def test_make_hetersynapse_constraint_modes():
    """Test different constraint modes in make_hetersynapse_constraint.

    This demonstrates the three constraint granularity options and
    visualizes the resulting number of constraint groups.
    """
    neurons = create_test_neurons(n_neurons=60)
    connections = create_test_connections(neurons, density=0.15)

    modes = ["full", "cell_only", "cell_and_receptor"]
    results = {}

    for mode in modes:
        constraint = make_hetersynapse_constraint(
            neurons,
            connections,
            cell_type_col="cell_type",
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            constraint_mode=mode,
        )

        # Get number of unique groups
        n_groups = int(constraint.data.max())
        results[mode] = n_groups

        assert scipy.sparse.issparse(constraint)
        assert constraint.nnz > 0  # Should have some constraints

    # Visualize constraint group counts
    fig, ax = plt.subplots(figsize=(8, 5))
    modes_list = list(results.keys())
    counts = [results[m] for m in modes_list]

    bars = ax.bar(modes_list, counts, color=["#3498db", "#e74c3c", "#2ecc71"])
    ax.set_ylabel("Number of Constraint Groups")
    ax.set_title("Constraint Groups by Mode")

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{count}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()
    save_fig(fig, "hetersynapse_constraint_modes")
    plt.close(fig)

    # Verify expected ordering: full > cell_and_receptor >= cell_only
    assert results["full"] >= results["cell_and_receptor"]
    assert results["cell_and_receptor"] >= results["cell_only"]

    print("✓ Constraint modes work correctly:")
    for mode, n_groups in results.items():
        print(f"  {mode:20s}: {n_groups} groups")


def test_make_hetersynapse_constrained_conn():
    """Test the convenience function that creates both conn and constraint.

    This demonstrates the recommended workflow for creating
    hetersynaptic connections with constraints ready for
    SparseConstrainedConn.
    """
    neurons = create_test_neurons(n_neurons=40)
    connections = create_test_connections(neurons, density=0.2)

    # Create both matrices in one call
    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="cell_only",  # Share weights across receptor types
    )

    # Verify outputs
    assert scipy.sparse.issparse(conn)
    assert scipy.sparse.issparse(constraint)
    assert isinstance(receptor_idx, pd.DataFrame)

    # Shapes should match
    assert conn.shape == constraint.shape

    # Constraint should have valid group IDs
    assert constraint.nnz > 0
    assert constraint.data.min() >= 1

    print("✓ make_hetersynapse_constrained_conn works correctly")
    print(f"  Connection shape: {conn.shape}")
    print(f"  Constraint groups: {int(constraint.data.max())}")


def test_hetersynapse_psc_get_psc_autodetection():
    """Test HeterSynapsePSC.get_psc with autodetection of receptor modes.

    This demonstrates how the mode is automatically detected from the
    receptor_type_index DataFrame structure.
    """
    neurons = create_test_neurons(n_neurons=30)
    connections = create_test_connections(neurons, density=0.2)

    # Create connection matrix
    conn_sp, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )

    n_receptor = len(receptor_idx)
    n_neurons = len(neurons)

    # Create linear layer
    linear = torch.nn.Linear(n_neurons, n_neurons * n_receptor, bias=False)

    with environ.context(dt=1.0):
        hetero_psc = HeterSynapsePSC(
            n_neuron=n_neurons,
            n_receptor=n_receptor,
            receptor_type_index=receptor_idx,
            linear=linear,
            base_psc=AlphaPSC,
            tau_syn=2.0,
        )

        init_net_state(hetero_psc, dtype=torch.float32)

        # Run a forward pass
        z = torch.randn(1, n_neurons)
        hetero_psc.single_step_forward(z)

    # Test autodetection with neuron mode (tuple input)
    if "pre_receptor_type" in receptor_idx.columns:
        # Get unique receptor types
        pre_types = receptor_idx["pre_receptor_type"].unique()
        post_types = receptor_idx["post_receptor_type"].unique()

        if len(pre_types) > 0 and len(post_types) > 0:
            # Try getting PSC by receptor pair
            pre_type = pre_types[0]
            post_type = post_types[0]

            # get_psc should use the psc from base_psc, pass it explicitly
            psc = hetero_psc.get_psc(
                receptor_type=(pre_type, post_type), psc=hetero_psc.base_psc.psc
            )
            assert psc is not None
            assert psc.shape[-1] == n_neurons

            print("✓ Autodetection works for neuron mode")
            print(f"  Retrieved PSC for ({pre_type}, {post_type})")

    # Get total PSC (None argument) - returns base_psc.psc with all receptor types
    psc_total = hetero_psc.get_psc(receptor_type=None)
    # The total PSC includes all receptor types, so shape is (n_neurons * n_receptor,)
    assert psc_total.shape[-1] == n_neurons * n_receptor

    print("✓ get_psc autodetection works correctly")


def test_sparse_constrained_conn_from_hetersynapse():
    """Test SparseConstrainedConn.from_hetersynapse class method.

    This demonstrates the clean workflow for creating constrained
    connections from hetersynapse data.
    """
    neurons = create_test_neurons(n_neurons=50)
    connections = create_test_connections(neurons, density=0.15)

    # Create heterosynapse connection and constraint
    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="full",
    )

    # Use the class method
    linear = SparseConstrainedConn.from_hetersynapse(
        conn=conn,
        constraint=constraint,
        receptor_type_index=receptor_idx,
        enforce_dale=True,
    )

    # Verify constraint_info is populated
    assert linear.constraint_info is not None
    assert "receptor_type_index" in linear.constraint_info
    pd.testing.assert_frame_equal(
        linear.constraint_info["receptor_type_index"], receptor_idx
    )

    print("✓ from_hetersynapse class method works correctly")
    print(f"  Magnitude shape: {linear.magnitude.shape}")


def test_sparse_constrained_conn_helper_methods():
    """Test SparseConstrainedConn helper methods for inspection and
    manipulation.

    This demonstrates:
    1. get_group_info() for inspecting constraint groups
    2. set_group_magnitude() for programmatic weight manipulation
    3. get_weights_by_group() for analysis
    """
    neurons = create_test_neurons(n_neurons=40)
    connections = create_test_connections(neurons, density=0.2)

    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="cell_and_receptor",
    )

    linear = SparseConstrainedConn.from_hetersynapse(
        conn, constraint, receptor_idx, enforce_dale=True
    )

    # Test get_group_info
    group_info = linear.get_group_info(include_weights=True)
    assert isinstance(group_info, pd.DataFrame)
    assert "group_id" in group_info.columns
    assert "num_connections" in group_info.columns
    assert "current_magnitude" in group_info.columns
    assert "mean_initial_weight" in group_info.columns

    print("✓ get_group_info works correctly")
    print(f"  Found {len(group_info)} constraint groups")

    # Test set_group_magnitude by group_id
    linear.set_group_magnitude(group_id=0, value=2.5)
    assert torch.isclose(linear.magnitude[0], torch.tensor(2.5))

    # Test get_weights_by_group
    weights_by_group = linear.get_weights_by_group()
    assert isinstance(weights_by_group, dict)
    assert len(weights_by_group) == len(linear.magnitude)

    # Visualize group sizes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Group sizes
    ax = axes[0]
    group_ids = group_info["group_id"].values
    num_conns = group_info["num_connections"].values
    ax.bar(group_ids, num_conns, color="#3498db")
    ax.set_xlabel("Group ID")
    ax.set_ylabel("Number of Connections")
    ax.set_title("Connections per Constraint Group")

    # Plot 2: Current magnitudes
    ax = axes[1]
    magnitudes = group_info["current_magnitude"].values
    ax.bar(group_ids, magnitudes, color="#e74c3c")
    ax.set_xlabel("Group ID")
    ax.set_ylabel("Magnitude")
    ax.set_title("Current Magnitude per Group")
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Initial (1.0)")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "constraint_group_info")
    plt.close(fig)

    print("✓ All helper methods work correctly")


def test_hetersynapse_workflow_example():
    """Complete workflow example demonstrating the full pipeline.

    This test serves as documentation for the recommended usage pattern.
    """
    # Step 1: Create test data
    neurons = create_test_neurons(n_neurons=60)
    connections = create_test_connections(neurons, density=0.15)

    # Step 2: Create hetersynapse connection with constraints
    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="cell_only",  # Share weights across receptor types
    )

    # Step 3: Initialize constrained connection
    linear = SparseConstrainedConn.from_hetersynapse(
        conn, constraint, receptor_idx, enforce_dale=True
    )

    # Step 4: Inspect constraint groups
    group_info = linear.get_group_info()
    print(f"Created {len(group_info)} constraint groups")

    # Step 5: (Optional) Manually adjust magnitudes
    # For example, boost E→I connections
    linear.set_group_magnitude(group_id=0, value=1.5)

    # Step 6: Use in forward pass
    n_neurons = len(neurons)
    x = torch.randn(10, n_neurons)  # batch_size=10
    y = linear(x)

    assert y.shape == (10, conn.shape[1])

    print("✓ Complete workflow example passed")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")


def test_stack_hetersynapse_dict():
    """Test converting dict format back to stacked matrix format.

    This demonstrates the round-trip conversion and allows for
    modification of individual receptor type matrices before stacking.
    """
    from btorch.connectome.connection import stack_hetersynapse

    neurons = create_test_neurons(n_neurons=40)
    connections = create_test_connections(neurons, density=0.15)

    # Get dict format
    conn_dict, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=True,
    )

    # Get stacked format for comparison
    conn_stacked_ref, receptor_idx_ref = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=False,
    )

    # Convert dict back to stacked
    conn_stacked = stack_hetersynapse(conn_dict, receptor_idx)

    # Should match the reference
    assert conn_stacked.shape == conn_stacked_ref.shape

    # Convert to same format for comparison
    conn_stacked = conn_stacked.tocsr()
    conn_stacked_ref = conn_stacked_ref.tocsr()

    # Check they're equal (data, indices, indptr)
    assert np.allclose(conn_stacked.data, conn_stacked_ref.data)
    assert np.array_equal(conn_stacked.indices, conn_stacked_ref.indices)
    assert np.array_equal(conn_stacked.indptr, conn_stacked_ref.indptr)

    print("✓ stack_hetersynapse_dict correctly converts dict to stacked format")

    # Test with modification
    conn_dict_modified = {}
    for k, v in conn_dict.items():
        # Convert to float for modification
        v_copy = v.copy()
        v_copy.data = v_copy.data.astype(float)
        conn_dict_modified[k] = v_copy

    # Scale E->I connections by 2.0
    if ("E", "I") in conn_dict_modified:
        conn_dict_modified[("E", "I")].data *= 2.0

    conn_stacked_modified = stack_hetersynapse(conn_dict_modified, receptor_idx)

    # The modified matrix should be different
    conn_stacked_modified = conn_stacked_modified.tocsr()
    assert not np.allclose(conn_stacked_modified.data, conn_stacked_ref.data)

    print("✓ Modifications to dict are preserved after stacking")
    print(f"  Dict keys: {list(conn_dict.keys())}")
    print(f"  Stacked shape: {conn_stacked.shape}")
