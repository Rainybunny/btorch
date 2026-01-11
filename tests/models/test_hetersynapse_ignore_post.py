import numpy as np
import pandas as pd
import torch

from btorch.connectome.connection import (
    make_hetersynapse_conn,
    make_hetersynapse_constrained_conn,
    stack_hetersynapse,
)
from btorch.models.linear import SparseConstrainedConn


def create_test_neurons(n_neurons=100, seed=42):
    np.random.seed(seed)
    neurons = pd.DataFrame(
        {
            "root_id": np.arange(1000, 1000 + n_neurons),
            "simple_id": np.arange(n_neurons),
        }
    )
    n_cell_types = 5
    neurons["cell_type"] = [f"type_{i % n_cell_types}" for i in range(n_neurons)]
    neurons["EI"] = np.random.choice(["E", "I"], size=n_neurons, p=[0.7, 0.3])
    return neurons


def create_test_connections(neurons, density=0.1, seed=42):
    np.random.seed(seed)
    n_neurons = len(neurons)
    n_connections = int(n_neurons**2 * density)
    pre_idx = np.random.choice(n_neurons, size=n_connections)
    post_idx = np.random.choice(n_neurons, size=n_connections)
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
    connections["pre_root_id"] = connections["pre_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )
    connections["post_root_id"] = connections["post_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )
    return connections


def test_hetersynapse_ignore_post_type():
    neurons = create_test_neurons(n_neurons=50)
    connections = create_test_connections(neurons, density=0.2)

    # Standard call
    conn_full, _ = make_hetersynapse_conn(
        neurons, connections, receptor_type_col="EI", ignore_post_type=False
    )
    # Expected: (N, 2N) assuming E and I pre-types, but wait,
    # Standard logic: product(pre, post) -> E-E, E-I, I-E, I-I -> 4 combinations
    # So columns = N * 4
    # Wait, existing code:
    # shape[1] * n_receptor_type
    # n_receptor_type = len(product(pre, post)) = 4
    # So N * 4.

    # Check what receptor types exist in data
    # We have E and I. so 2 types.
    # Product is 2*2 = 4 types.

    # Ignore post type call
    conn_ignore, receptor_idx = make_hetersynapse_conn(
        neurons, connections, receptor_type_col="EI", ignore_post_type=True
    )

    # Verify shape
    n_neurons = len(neurons)
    # unique pre types: E, I -> 2 types
    assert conn_ignore.shape == (n_neurons, n_neurons * 2)

    # Verify receptor_idx structure
    assert "receptor_type" in receptor_idx.columns
    assert "pre_receptor_type" not in receptor_idx.columns
    assert len(receptor_idx) == 2

    print("✓ Shape reduction confirmed")


def test_hetersynapse_constrained_ignore_post_type():
    neurons = create_test_neurons(n_neurons=50)
    connections = create_test_connections(neurons, density=0.2)

    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        constraint_mode="full",
        ignore_post_type=True,
    )

    assert conn.shape == constraint.shape
    assert conn.shape[1] == len(neurons) * 2

    # Check constrained connection initialization
    layer = SparseConstrainedConn.from_hetersynapse(
        conn, constraint, receptor_idx, enforce_dale=True
    )

    # Should work without error
    x = torch.randn(10, len(neurons))
    y = layer(x)
    assert y.shape == (10, len(neurons) * 2)

    print("✓ Constrained connection works with ignore_post_type")


def test_dict_roundtrip_ignore_post_type():
    neurons = create_test_neurons(n_neurons=40)
    connections = create_test_connections(neurons, density=0.15)

    conn_dict, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        return_dict=True,
        ignore_post_type=True,
    )

    assert isinstance(conn_dict, dict)
    # Keys should be just pre-synaptic types (strings)
    assert set(conn_dict.keys()) == {"E", "I"} or set(conn_dict.keys()) == {"I", "E"}

    # Stack back
    conn_stacked = stack_hetersynapse(conn_dict, receptor_idx)

    conn_ref, _ = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        return_dict=False,
        ignore_post_type=True,
    )

    # Compare
    conn_stacked = conn_stacked.tocsr()
    conn_ref = conn_ref.tocsr()

    assert np.allclose(conn_stacked.data, conn_ref.data)
    assert np.array_equal(conn_stacked.indices, conn_ref.indices)
    assert np.array_equal(conn_stacked.indptr, conn_ref.indptr)

    print("✓ Dict roundtrip works with ignore_post_type")


if __name__ == "__main__":
    test_hetersynapse_ignore_post_type()
    test_hetersynapse_constrained_ignore_post_type()
    test_dict_roundtrip_ignore_post_type()
