"""Benchmark spiking analysis functions with large inputs.

Tests performance of CV, Fano factor, Kurtosis, Local Variation, ECI, and
lag correlation with realistic large-scale inputs: T=1000, B=8, N=100000
"""

import time

import torch

from btorch.analysis.dynamic_tools.ei_balance import (
    compute_eci,
    compute_ei_balance,
    compute_lag_correlation,
)
from btorch.analysis.spiking import (
    fano,
    isi_cv,
    kurtosis,
    local_variation,
)


def generate_spike_data(T, B, N, rate=0.05, device="cpu", dtype=torch.float32):
    """Generate synthetic spike data.

    Args:
        T: Time steps
        B: Batch size (trials)
        N: Number of neurons
        rate: Firing probability per time step
        device: torch device
        dtype: torch dtype

    Returns:
        spike_data: Tensor of shape [T, B, N]
    """
    torch.manual_seed(42)
    spikes = (torch.rand(T, B, N, device=device, dtype=torch.float32) < rate).to(dtype)
    return spikes


def generate_current_data(T, B, N, device="cpu", dtype=torch.float32):
    """Generate synthetic current data for E/I balance analysis.

    Args:
        T: Time steps
        B: Batch size (trials)
        N: Number of neurons
        device: torch device
        dtype: torch dtype

    Returns:
        I_e, I_i: Excitatory and inhibitory currents [T, B, N]
    """
    torch.manual_seed(42)
    I_e = torch.randn(T, B, N, device=device, dtype=dtype) * 0.5 + 1.0
    I_i = -torch.randn(T, B, N, device=device, dtype=dtype) * 0.5 - 0.5
    return I_e, I_i


def benchmark_cv(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark ISI CV computation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking CV: T={T}, B={B}, N={N}, device={device}, dtype={dtype}")
    print(f"{'='*60}")

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)

    # Warmup
    _ = isi_cv(spikes, dt_ms=1.0, batch_axis=(1,))

    # Benchmark
    start = time.time()
    cv_values, info = isi_cv(spikes, dt_ms=1.0, batch_axis=(1,))
    elapsed = time.time() - start

    print(f"  Shape: {spikes.shape} -> CV shape: {cv_values.shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M spikes/sec")

    return elapsed


def benchmark_fano(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark Fano factor computation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Fano: T={T}, B={B}, N={N}, device={device}, dtype={dtype}")
    print(f"{'='*60}")

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)

    # Warmup
    _ = fano(spikes, window=100, batch_axis=(1,))

    # Benchmark
    start = time.time()
    fano_values, info = fano(spikes, window=100, batch_axis=(1,))
    elapsed = time.time() - start

    print(f"  Shape: {spikes.shape} -> Fano shape: {fano_values.shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M spikes/sec")

    return elapsed


def benchmark_kurtosis(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark Kurtosis computation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Kurtosis: T={T}, B={B}, N={N}, device={device}, dtype={dtype}")
    print(f"{'='*60}")

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)

    # Warmup
    _ = kurtosis(spikes, window=100, batch_axis=(1,))

    # Benchmark
    start = time.time()
    kurt_values, info = kurtosis(spikes, window=100, batch_axis=(1,))
    elapsed = time.time() - start

    print(f"  Shape: {spikes.shape} -> Kurt shape: {kurt_values.shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M spikes/sec")

    return elapsed


def benchmark_lv(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark Local Variation computation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking LV: T={T}, B={B}, N={N}, device={device}, dtype={dtype}")
    print(f"{'='*60}")

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)

    # Warmup
    _ = local_variation(spikes, dt_ms=1.0, batch_axis=(1,))

    # Benchmark
    start = time.time()
    lv_values, info = local_variation(spikes, dt_ms=1.0, batch_axis=(1,))
    elapsed = time.time() - start

    print(f"  Shape: {spikes.shape} -> LV shape: {lv_values.shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M spikes/sec")

    return elapsed


def benchmark_eci(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark ECI computation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking ECI: T={T}, B={B}, N={N}, device={device}, dtype={dtype}")
    print(f"{'='*60}")

    I_e, I_i = generate_current_data(T, B, N, device=device, dtype=dtype)

    # Warmup
    _ = compute_eci(I_e, I_i, batch_axis=(1,))

    # Benchmark
    start = time.time()
    eci, info = compute_eci(I_e, I_i, batch_axis=(1,))
    elapsed = time.time() - start

    print(f"  Shape: {I_e.shape} -> ECI shape: {eci.shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M values/sec")

    return elapsed


def benchmark_lag_correlation(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark lag correlation computation."""
    print(f"\n{'='*60}")
    print(
        f"Benchmarking Lag Correlation: T={T}, B={B}, N={N}, "
        f"device={device}, dtype={dtype}"
    )
    print(f"{'='*60}")

    I_e, I_i = generate_current_data(T, B, N, device=device, dtype=dtype)

    # Warmup
    _ = compute_lag_correlation(I_e, -I_i, dt=1.0, max_lag_ms=30.0, batch_axis=(1,))

    # Benchmark
    start = time.time()
    peak_corr, best_lag_ms, info = compute_lag_correlation(
        I_e, -I_i, dt=1.0, max_lag_ms=30.0, batch_axis=(1,)
    )
    elapsed = time.time() - start

    print(f"  Shape: {I_e.shape} -> Corr shape: {peak_corr.shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M values/sec")

    return elapsed


def benchmark_ei_full(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark full E/I balance computation."""
    print(f"\n{'='*60}")
    print(
        f"Benchmarking E/I Balance Full: T={T}, B={B}, N={N}, "
        f"device={device}, dtype={dtype}"
    )
    print(f"{'='*60}")

    I_e, I_i = generate_current_data(T, B, N, device=device, dtype=dtype)

    # Warmup
    _ = compute_ei_balance(I_e, I_i, batch_axis=(1,))

    # Benchmark
    start = time.time()
    eci, peak_corr, best_lag_ms, info = compute_ei_balance(I_e, I_i, batch_axis=(1,))
    elapsed = time.time() - start

    print(f"  Shape: {I_e.shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M values/sec")
    print(
        f"  Results: eci_mean={eci.mean():.3f}, "
        f"peak_corr_mean={peak_corr.mean():.3f}"
    )

    return elapsed


def compare_dtypes(T=1000, B=8, N=100000, device="cpu"):
    """Compare float16 vs float32 performance."""
    print(f"\n{'='*60}")
    print(f"Comparing dtypes: T={T}, B={B}, N={N}, device={device}")
    print(f"{'='*60}")

    for dtype in [torch.float16, torch.float32]:
        spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)
        mem_mb = spikes.numel() * spikes.element_size() / 1024 / 1024

        start = time.time()
        _ = fano(spikes, window=100, batch_axis=(1,))
        elapsed = time.time() - start

        print(f"  {dtype}: {mem_mb:.1f}MB, {elapsed:.3f}s")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("Spiking Analysis Benchmarks")
    print("=" * 60)

    T, B, N = 1000, 8, 100000

    # CPU benchmarks
    print("\n" + "=" * 60)
    print("CPU BENCHMARKS")
    print("=" * 60)

    benchmark_cv(T, B, N, device="cpu")
    benchmark_fano(T, B, N, device="cpu")
    benchmark_kurtosis(T, B, N, device="cpu")
    benchmark_lv(T, B, N, device="cpu")

    # E/I balance benchmarks
    print("\n" + "=" * 60)
    print("E/I BALANCE BENCHMARKS (CPU)")
    print("=" * 60)

    benchmark_eci(T, B, N, device="cpu")
    benchmark_lag_correlation(T, B, N, device="cpu")
    benchmark_ei_full(T, B, N, device="cpu")

    # GPU benchmarks if available
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU BENCHMARKS")
        print("=" * 60)

        benchmark_cv(T, B, N, device="cuda")
        benchmark_fano(T, B, N, device="cuda")
        benchmark_kurtosis(T, B, N, device="cuda")
        benchmark_lv(T, B, N, device="cuda")

        # E/I balance GPU benchmarks
        print("\n" + "=" * 60)
        print("E/I BALANCE BENCHMARKS (GPU)")
        print("=" * 60)

        benchmark_eci(T, B, N, device="cuda")
        benchmark_lag_correlation(T, B, N, device="cuda")
        benchmark_ei_full(T, B, N, device="cuda")

        # Compare dtypes on GPU
        compare_dtypes(T, B, N, device="cuda")
    else:
        print("\nCUDA not available, skipping GPU benchmarks")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
