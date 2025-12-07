from typing import Literal

import numpy as np
from scipy.signal import welch


def describe_array(array: np.ndarray):
    """Print descriptive statistics for a 1D array."""
    mean = np.mean(array)
    median = np.median(array)
    std_dev = np.std(array)
    min_val = np.min(array)
    max_val = np.max(array)
    q25 = np.percentile(array, 25)
    q50 = np.percentile(array, 50)  # This is the same as the median
    q75 = np.percentile(array, 75)

    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"25th Percentile (Q1): {q25}")
    print(f"50th Percentile (Q2/Median): {q50}")
    print(f"75th Percentile (Q3): {q75}")


def compute_log_hist(data, bins=1000, edge_pos: Literal["mid", "sep"] = "mid"):
    bin_edges = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), num=bins)
    hist, edges = np.histogram(data, bins=bin_edges)
    if edge_pos == "mid":
        bin_edges = 0.5 * (edges[:-1] + edges[1:])
    return hist, bin_edges


def compute_spectrum(y, dt, nperseg=None):
    freqs, Y_mag = welch(y, fs=1 / dt, nperseg=nperseg, axis=0)
    return freqs, Y_mag
