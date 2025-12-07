import numpy as np
import pandas as pd
import scipy.sparse

from btorch.analysis.aggregation import agg_by_neuron, agg_conn


def test_agg_by_neuron_groups_by_cell_type():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    neurons = pd.DataFrame(
        {
            "simple_id": [0, 1, 2],
            "cell_type": ["A", "B", "A"],
        }
    )

    result = agg_by_neuron(y, neurons, agg="mean")
    np.testing.assert_allclose(result["A"], np.array([2.0, 5.0]))
    np.testing.assert_allclose(result["B"], np.array([2.0, 5.0]))


def test_agg_conn_with_sparse_weights_by_neuropil():
    conn = pd.DataFrame(
        {
            "pre_simple_id": [0, 1],
            "post_simple_id": [1, 0],
            "neuropil": ["alpha", "alpha"],
        }
    )
    weights = scipy.sparse.coo_array(
        (
            np.array([0.5, 1.5]),
            (np.array([0, 1]), np.array([1, 0])),
        ),
        shape=(2, 2),
    )

    aggregated = agg_conn(
        y=np.array([]), conn=conn, conn_weight=weights, mode="neuropil", agg="sum"
    )

    np.testing.assert_allclose(aggregated.loc["alpha"], 2.0)
