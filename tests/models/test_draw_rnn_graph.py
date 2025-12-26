import pytest
import torch

from btorch.models.rnn import make_rnn
from btorch.utils.file import fig_path
from tests.models.test_rnn import last_step_sum, SimpleRNNCell


def test_draw_rnn_graph():
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    try:
        from torchviz import make_dot
    except ImportError:
        make_dot = None

    torch.manual_seed(233)

    rnn = make_rnn(
        SimpleRNNCell,
        grad_checkpoint=False,
        unroll=False,
        save_grad_history=True,
        grad_state_names=["h"],
        update_state_names=["h"],
        allow_buffer=True,
    )(input_size=3, hidden_size=4)

    T, batch_size, input_size = 20, 2, 3
    x = torch.ones(T, batch_size, input_size, requires_grad=True)

    rnn.reset()
    out, _ = rnn(x)
    loss = last_step_sum(out)
    loss.backward()

    if make_dot is not None:
        grad_graph = make_dot(loss, params=dict(rnn.named_parameters()))
        grad_graph.render(fig_path() / "rnn_grad_graph", format="png")

    grad_history = rnn.get_grad_history()
    rnn.clear_grad_history()

    grad_magnitudes = []
    for g in grad_history["h"]:
        grad_magnitudes.append(0.0 if g is None else g.norm().item())

    h_magnitudes = [out[t].norm().item() for t in range(out.shape[0])]

    plt.figure(figsize=(6, 4))
    plt.plot(grad_magnitudes, marker="o", label="||grad h_t||")
    plt.plot(h_magnitudes, marker="s", label="||h_t||")
    plt.title("Hidden State and Gradient Magnitudes Across Time")
    plt.xlabel("Time step")
    plt.ylabel("L2 norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = fig_path() / "grad_h_timeseries.png"
    plt.savefig(save_path)
    plt.close()
