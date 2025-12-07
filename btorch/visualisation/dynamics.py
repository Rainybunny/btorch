import numpy as np
import pandas as pd

from ..analysis.spiking import cv_from_spikes


def plot_cv_distribution(result, model_param, args, root_id_converter):
    """Plot CV distribution and statistics."""
    import matplotlib.pyplot as plt

    spike_data = result["neuron"]["spike"]
    dt_ms = model_param["dt"]

    # Calculate CV values
    cv_values, isi_stats = cv_from_spikes(spike_data, dt_ms)

    # Filter out NaN values for plotting
    valid_cv = cv_values[~np.isnan(cv_values)]

    if len(valid_cv) == 0:
        print("No valid CV values found (need at least 2 spikes per neuron)")
        return None

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot CV distribution histogram
    ax1.hist(valid_cv, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Coefficient of Variation (CV)")
    ax1.set_ylabel("Number of Neurons")
    ax1.set_title(f"CV Distribution (n={len(valid_cv)} neurons)")
    ax1.axvline(
        np.mean(valid_cv),
        color="red",
        linestyle="--",
        label=f"Mean CV = {np.mean(valid_cv):.3f}",
    )
    ax1.axvline(1.1, color="green", linestyle="--", label="Target CV = 1.1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot CV vs firing rate scatter
    firing_rates = np.mean(spike_data, axis=0) * (1000.0 / dt_ms)  # Convert to Hz
    valid_mask = ~np.isnan(cv_values)
    valid_fr = firing_rates[valid_mask]
    valid_cv_for_scatter = cv_values[valid_mask]

    ax2.scatter(valid_fr, valid_cv_for_scatter, alpha=0.6)
    ax2.set_xlabel("Firing Rate (Hz)")
    ax2.set_ylabel("Coefficient of Variation (CV)")
    ax2.set_title("CV vs Firing Rate")
    ax2.axhline(1.1, color="green", linestyle="--", label="Target CV = 1.1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{args.figure_dir}/cv_distribution.pdf", bbox_inches="tight")

    # Save CV statistics to CSV
    cv_df = pd.DataFrame(
        {
            "neuron_idx": np.arange(len(cv_values)),
            "cv": cv_values,
            "firing_rate_hz": firing_rates,
            "n_spikes": [isi_stats[i]["n_spikes"] for i in range(len(cv_values))],
            "mean_isi_ms": [isi_stats[i]["mean_isi"] for i in range(len(cv_values))],
            "std_isi_ms": [isi_stats[i]["std_isi"] for i in range(len(cv_values))],
        }
    )

    # Add root IDs if converter is available
    if root_id_converter is not None:
        cv_df["root_id"] = root_id_converter(np.arange(len(cv_values)))

    cv_df.to_csv(f"{args.figure_dir}/cv_statistics.csv", index=False)

    # Print summary statistics
    print("\nCV Statistics Summary:")
    print(f"Total neurons: {len(cv_values)}")
    print(f"Neurons with valid CV: {len(valid_cv)}")
    print(f"Mean CV: {np.mean(valid_cv):.3f} ± {np.std(valid_cv):.3f}")
    print(f"Median CV: {np.median(valid_cv):.3f}")
    print(f"CV range: [{np.min(valid_cv):.3f}, {np.max(valid_cv):.3f}]")
    print(f"Neurons with CV ≈ 1.1 (±0.2): {np.sum(np.abs(valid_cv - 1.1) <= 0.2)}")

    return fig, cv_df
