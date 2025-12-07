from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.axes import Axes

from ..analysis.spiking import firing_rate, raster_plot
from ..analysis.statistics import compute_log_hist, compute_spectrum
from .braintools_compat import get_figure


def plot_population_by_field(
    res,
    field: str,
    ts=None,
    name=None,
    population: Optional[Sequence[int]] = None,
    title_format: Callable[[Sequence[Any]], Any] = lambda arg: f"[{arg[0]}] = {arg[1]}",
    ref_line=None,
    ref_name=None,
    plot_args=None,
):
    if name is None:
        name = field
    fields = field.split(".")
    if len(fields) > 1:
        while len(fields) > 1:
            field = fields.pop(0)
            res = res[field]
    else:
        res = res[field]
    n_ts = res.shape[0]
    if ts is None:
        ts = np.arange(n_ts)
    if population is None:
        num = res.shape[1]
        n_sel = 10 if num > 10 else num
        population = np.random.choice(n_sel, 10, replace=False)
    n_sel = len(population)
    fig, gs = get_figure(
        n_sel,
        1,
        1.5,
        8,
    )
    if res.ndim == 3:
        n_lines = res.shape[2]
        if plot_args is None:
            plot_args = [None, None]
        assert len(plot_args) == n_lines
    elif res[field].ndim == 2:
        n_lines = 1
    else:
        raise ValueError("res cannot contain higher dim traces")
    if n_lines == 1:
        if isinstance(plot_args, Sequence) and plot_args is not None:
            assert len(plot_args) == 1
            plot_args = plot_args[0]

    fig.suptitle(f"{name}", fontsize=16)
    for i, s in enumerate(population):
        ax = fig.add_subplot(gs[i, 0])
        if n_lines > 1:
            for j in range(res[field][:, s].shape[-1]):
                ax.plot(ts, res[field][:, s, j], label=f"{field}-{j}", **plot_args[j])
        elif n_lines == 1:
            ax.plot(ts, res[field][:, s], label=f"{field}", **plot_args)
        if ref_line is not None:
            if ref_name is None:
                ref_name = f"{field}_ref"
            ref_l = (
                ref_line[s]
                if hasattr(ref_line, "size") and ref_line.size == n_ts
                else ref_line
            )
            ax.hlines(ref_l, ts[0], ts[-1], label=ref_name)
        ax.set_xlabel("Time (ms)")
        ax.set_xlim(-0.1, ts[-1] + 0.1)
        ax.set_title(title_format((i, s)))
        ax.legend(loc="upper right")
    return fig


def plot_population_by_field_single_plot(
    res,
    field: str,
    ts=None,
    name=None,
    population: np.ndarray | Sequence[int] | int | None = None,
    legend_format: Optional[
        Callable[[Sequence[Any]], Any]
    ] = lambda arg: f"[{arg[0]}] = {arg[1]}",
    ref_line: Optional[float | Sequence[float]] = None,
    ref_name: Optional[Union[str, Sequence[str]]] = None,
    plot_args: Optional[Union[dict[str, Any], Sequence[dict[str, Any]]]] = None,
    color_map: Optional[Sequence[Any]] = None,
):
    """Plot all traces recorded in res['field'] in one figure."""

    fields = field.split(".")
    if len(fields) > 1:
        while len(fields) > 0:
            field = fields.pop(0)
            res = res[field]
    else:
        res = res[field]
    data = res
    num_dims = len(data.shape)
    if name is None:
        name = field

    if ts is None:
        ts = np.arange(data.shape[0])

    if not isinstance(population, (Sequence, np.ndarray, torch.Tensor)):
        num = res.shape[1]
        if population is None:
            n_sel = 10 if num > 10 else num
        elif isinstance(population, int):
            n_sel = population
        else:
            raise ValueError(f"Unexpected population type: {type(population)}")

        population = np.random.choice(num, n_sel, replace=False)

    if color_map is None:
        cmap = cm.get_cmap("turbo", len(population))
        color_map = [cmap(j) for j in range(len(population))]
    else:
        assert len(color_map) == len(population)

    assert (
        ts is None or len(ts) == res.shape[0]
    ), "ts must have the same length as the first dimension of res[field]"

    if isinstance(plot_args, Sequence) and num_dims == 3:
        assert len(plot_args) == data.shape[2]
    if plot_args is None:
        plot_args = [{}]
    elif isinstance(plot_args, dict):
        plot_args = [plot_args] * (data.shape[2] if num_dims == 3 else 1)
    elif num_dims == 2:
        plot_args = [plot_args]

    if isinstance(ref_line, Sequence) and num_dims == 3:
        assert len(ref_line) == data.shape[2]
    if isinstance(ref_line, (int, float)) or ref_line is None:
        ref_line = [ref_line] * (data.shape[2] if num_dims == 3 else 1)
    if isinstance(ref_name, Sequence) and num_dims == 3:
        assert len(ref_name) == data.shape[2]
    if isinstance(ref_name, str) or ref_name is None:
        ref_name = [ref_name] * (data.shape[2] if num_dims == 3 else 1)

    figs = []
    for i in range(data.shape[2]) if num_dims == 3 else (0,):
        fig, ax = plt.subplots()
        for j, p in enumerate(population):
            ax.plot(
                ts,
                data[:, p, i] if num_dims == 3 else data[:, p],
                label=legend_format([j, p]) if legend_format is not None else None,
                color=color_map[j],
                alpha=0.8,
                lw=0.5,
                **plot_args[i],
            )
        if ref_line[i] is not None:
            ax.axhline(ref_line[i], label=ref_name[i], linewidth=2, color="black")
        if legend_format is not None or ref_name[i] is not None:
            ax.legend(ncol=len(population) // 16, bbox_to_anchor=(1, 1))
        if num_dims == 3:
            ax.set_title(f"{name} - {i}")
        else:
            ax.set_title(name)
        ax.set_xlabel("Time (ms)")
        figs.append(fig)

    return figs, color_map


def plot_activity(sp_matrix, times=None):
    if times is None:
        times = np.arange(sp_matrix.shape[0])
    fig, gs = get_figure(3, 1, col_len=12)
    index, ts = raster_plot(sp_matrix, times)
    ax0 = plt.subplot(gs[:-1])
    spiked = np.unique(index).size
    ax0.set_title(
        f"total number of neurons that spiked: {spiked}/{sp_matrix.shape[-1]}"
    )
    ax0.set_xlim(*times[[0, -1]])
    ax0.set_ylim(0, sp_matrix.shape[1])
    ax0.set_ylabel("Neuron Index")
    ax0.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax0.scatter(ts, index, s=0.5)
    fr = firing_rate(sp_matrix, 10.0, 1.0e-3)
    ax1 = plt.subplot(gs[-1])
    ax1.plot(times, fr)
    ax1.set_ylabel("Firing rate (Hz)")
    ax1.set_xlabel("Time [ms]")
    ax1.set_xlim(*times[[0, -1]])
    skip_t = len(fr) // 8
    if skip_t < 100:
        skip_t = 0
    ax1.set_ylim(0, fr[skip_t:].max() * 1.1)
    return fig, [ax0, ax1]


def plot_firing_rate_spectrum(
    mean_rate,
    dt=None,
    times=None,
    ax=None,
    mode: str = "loglog",
    show_mean: bool = False,
    nperseg: Optional[int] = None,
    title: str = "Firing Rate Spectrum",
):
    """Compute and plot the frequency spectrum of the mean firing rate."""
    if dt is None and times is not None:
        dt = np.mean(np.diff(times))
    if dt is None and times is None:
        dt = 1.0

    freqs, power = compute_spectrum(mean_rate, dt=dt, nperseg=nperseg)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    power_db = 10 * np.log10(power)

    if mode == "db_logx":
        if not show_mean:
            ax.semilogx(freqs, power_db, color="b")
        else:
            ax.semilogx(freqs, power_db, color="b", alpha=0.25, lw=0.5)
            ax.semilogx(freqs, power_db.mean(axis=1), color="k", lw=1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB/Hz)")
    elif mode == "db":
        if not show_mean:
            ax.plot(freqs, power_db, color="b")
        else:
            ax.plot(freqs, power_db, color="b", alpha=0.25, lw=0.5)
            ax.plot(freqs, power_db.mean(axis=1), color="k", lw=1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB/Hz)")
    elif mode == "loglog":
        if not show_mean:
            ax.loglog(freqs, power, color="b")
        else:
            ax.loglog(freqs, power, color="b", alpha=0.25, lw=0.5)
            ax.loglog(freqs, power.mean(axis=1), color="k", lw=1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (a.u.)")
    else:
        raise ValueError(f"Unsupported mode {mode}")

    ax.set_title(title)

    return power, freqs, fig, ax


def plot_hist_dist_loglog(weights, name, ax: Optional[Axes] = None, **plot_kargs):
    hist, bin_centers = compute_log_hist(weights)

    fig = None
    default_axis = False
    if ax is None:
        default_axis = True
        fig, ax = plt.subplots()

    ax.scatter(bin_centers, hist, s=2, **plot_kargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if default_axis:
        ax.set_xlabel(f"{name}")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {name}")

    return fig, ax
