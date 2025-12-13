import matplotlib.pyplot as plt
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.izhikevich import Izhikevich

from ...utils.file import save_fig


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DT = 1.0
T = 400
TIME = torch.arange(0, T, DT)

# Parameter sets follow the generalized Izhikevich model (C, k, v_r, v_t,
# v_peak, a, b, c, d)
PHENOTYPES = [
    {
        "name": "regular_spiking",
        "title": "Regular Spiking (RS)",
        "params": {"a": 0.03, "b": -2.0, "v_reset": -50.0, "d": 100.0},
        "stimulus": lambda steps: torch.cat(
            (torch.zeros(50), torch.full((steps - 50,), 7.0))
        ),
    },
    {
        "name": "phasic_spiking",
        "title": "Phasic Spiking (PhS)",
        "params": {"a": 0.02, "b": 0.25, "v_reset": -65.0, "d": 2.0},
        "stimulus": lambda steps: torch.cat(
            (torch.full((40,), 20.0), torch.zeros((steps - 40,)))
        ),
    },
    {
        "name": "fast_spiking",
        "title": "Fast Spiking (FS)",
        "params": {"a": 0.1, "b": 2.0, "v_reset": -65.0, "d": 2.0},
        "stimulus": lambda steps: torch.cat(
            (torch.zeros(50), torch.full((steps - 50,), 20.0))
        ),
    },
    {
        "name": "chattering",
        "title": "Chattering (CH)",
        "params": {"a": 0.03, "b": 1.0, "v_reset": -40.0, "d": 150.0},
        "stimulus": lambda steps: torch.cat(
            (torch.zeros(50), torch.full((steps - 50,), 10.0))
        ),
    },
    {
        "name": "low_threshold_spiking",
        "title": "Low-Threshold Spiking (LTS)",
        "params": {"a": 0.03, "b": 8.0, "v_reset": -55.0, "d": 150.0},
        "stimulus": lambda steps: torch.cat(
            (torch.zeros(50), torch.full((steps - 50,), 3.0))
        ),
    },
    {
        "name": "intrinsically_bursting",
        "title": "Intrinsically Bursting (IB)",
        "params": {"a": 0.01, "b": 5.0, "v_reset": -56.0, "d": 130.0},
        "stimulus": lambda steps: torch.cat(
            (torch.zeros(50), torch.full((steps - 50,), 10.0))
        ),
    },
    {
        "name": "thalamo_cortical",
        "title": "Thalamo-Cortical (TC)",
        "params": {"a": 0.02, "b": -0.5, "v_reset": -65.0, "d": 100.0},
        "stimulus": lambda steps: torch.cat(
            (torch.zeros(50), torch.full((steps - 50,), 2.5))
        ),
    },
    {
        "name": "rebound_spiking",
        "title": "Rebound Spiking (ReS)",
        "params": {"a": 0.03, "b": 2.0, "v_reset": -60.0, "d": 100.0},
        "stimulus": lambda steps: torch.cat(
            (torch.full((80,), -15.0), torch.zeros((steps - 80,)))
        ),
    },
]


def _simulate(neuron: Izhikevich, stimulus: torch.Tensor):
    traces = {"spike": [], "v": [], "u": []}
    with torch.no_grad():
        with environ.context(dt=float(DT)):
            for current in stimulus:
                spike = neuron(current.expand(neuron.n_neuron))
                traces["spike"].append(spike.detach().cpu())
                traces["v"].append(neuron.v.detach().cpu())
                traces["u"].append(neuron.u.detach().cpu())

    traces = {k: torch.stack(v) for k, v in traces.items()}
    return traces


def _plot_grid(time, results):
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(12, 3 * n), sharex=True)

    for idx, case_result in enumerate(results):
        case, traces = case_result["case"], case_result["traces"]
        spikes = traces["spike"].squeeze(-1)
        v = traces["v"].squeeze(-1)
        u = traces["u"].squeeze(-1)

        firing_rate = spikes.sum() / len(time) * 1000

        ax_spk, ax_v, ax_u = axes[idx]
        spk_nz = spikes.nonzero(as_tuple=False)
        if len(spk_nz) > 0:
            ax_spk.scatter(time[spk_nz[:, 0]], spk_nz[:, 1], marker="|", linewidths=0.8)
        ax_spk.set_ylabel("Spikes")
        ax_spk.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_spk.set_title(f"{case['title']} — {firing_rate.item():.1f} Hz")

        ax_v.plot(time, v, linewidth=0.9)
        ax_v.axhline(
            y=case_result["v_threshold"], color="r", linestyle="--", label="Threshold"
        )
        ax_v.set_ylabel("Membrane Potential")

        ax_u.plot(time, u, linewidth=0.9)
        ax_u.set_ylabel("Recovery (u)")
        ax_u.set_xlabel("Time (ms)")

    handles, labels = axes[0][1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Izhikevich Classic Phenotypes", fontsize=12)
    fig.tight_layout()
    save_fig(fig, name="izhikevich_phenotypes_grid")
    plt.close(fig)


def test_izhikevich_classic_phenotypes():
    base_params = {
        "n_neuron": 1,
        "v_threshold": 35.0,
        "v_reset": -60.0,
        "v_rest": -60.0,
        "c_m": 100.0,
        "k": 0.7,
        "v_T": -40.0,
        "detach_reset": False,
        "step_mode": "s",
        "backend": "torch",
        "device": DEVICE,
    }

    results = []
    for case in PHENOTYPES:
        params = base_params.copy()
        params.update(case["params"])

        neuron = Izhikevich(**params)
        init_net_state(neuron, device=DEVICE)

        stimulus = case["stimulus"](len(TIME)).to(device=DEVICE, dtype=torch.float32)
        traces = _simulate(neuron, stimulus)
        results.append(
            {
                "case": case,
                "traces": traces,
                "v_threshold": params["v_threshold"],
            }
        )

    _plot_grid(TIME, results)
