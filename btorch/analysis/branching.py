# WiltingPriesemann toolbox for MR estimation
# Authors: Jens Wilting & Viola Priesemann
# Minimal toolbox for MR estimation according to Wilting & Priesemann, 2018

import numpy as np


def simulate_branching(length=10000, m=0.9, activity=100, rng=None):
    """Simulate branching process with immigration."""
    rng = np.random.default_rng(rng)
    h = activity * (1 - m)
    A_t = np.empty(length, dtype=np.int64)
    A_t[0] = rng.poisson(lam=activity)
    for t in range(1, length):
        tmp = rng.poisson(lam=h)
        if A_t[t - 1] > 0 and m > 0:
            tmp += rng.poisson(lam=m * A_t[t - 1])
        A_t[t] = tmp
    return A_t


def simulate_binomial_subsampling(A_t, alpha, rng=None):
    """Binomial subsampling: each count is sampled with prob alpha."""
    rng = np.random.default_rng(rng)
    A_t = np.asarray(A_t, dtype=np.int64)
    out = rng.binomial(A_t, alpha)
    return out.astype(float)


def input_handler(items):
    if isinstance(items, np.ndarray):
        if items.ndim == 1 and items.dtype.kind in "iuf":
            return [items]
        if items.ndim == 1 and items.dtype.kind == "S":
            return [np.load(items)]
        if items.ndim == 2 and items.dtype.kind in "iuf":
            return [row for row in items]
        raise ValueError("Unsupported numpy array input")
    if isinstance(items, list):
        if all(isinstance(x, (np.ndarray, list)) for x in items):
            return [np.asarray(x) for x in items]
        if all(isinstance(x, str) for x in items):
            return [np.load(x) for x in items]
        raise ValueError("List of mixed types not supported")
    if isinstance(items, str):
        return [np.load(items)]
    raise ValueError("Input type not recognized")


def get_slopes(all_counts, k_max, min_points=50, scatterpoints=None, eps=1e-12):
    """Compute r_k robustly for k = 1 .."""
    k_arr = np.arange(1, k_max, dtype=int)
    r_k = np.zeros(len(k_arr), dtype=np.float64)
    stderr = np.zeros(len(k_arr), dtype=np.float64)

    xs = []
    ys = []

    for idx, k in enumerate(k_arr):
        x_concat = []
        y_concat = []
        for counts in all_counts:
            counts = np.asarray(counts, dtype=np.float64)
            if counts.size <= k:
                continue
            x_block = counts[:-k]
            y_block = counts[k:]
            x_block = x_block - x_block.mean()
            y_block = y_block - y_block.mean()
            x_concat.append(x_block)
            y_concat.append(y_block)

        if not x_concat:
            r_k[idx] = np.nan
            stderr[idx] = np.nan
            continue

        x = np.concatenate(x_concat)
        y = np.concatenate(y_concat)

        N = x.size
        if N < min_points:
            r_k[idx] = np.nan
            stderr[idx] = np.nan
            continue

        cov = np.mean(x * y)
        var0 = np.mean(x * x)
        if var0 <= eps:
            r_k[idx] = np.nan
            stderr[idx] = np.nan
            continue

        slope = cov / var0
        prod = x * y
        stderr_prod = prod.std(ddof=1) / np.sqrt(N)
        stderr_slope = stderr_prod / (var0 + eps)

        r_k[idx] = slope
        stderr[idx] = stderr_slope

        if scatterpoints:
            xs.append(x.copy())
            ys.append(y.copy())

    valid_idx = np.where(~np.isnan(r_k))[0]
    if valid_idx.size:
        first = valid_idx[0]
        data_length = int(np.sum([len(a) for a in all_counts]) - k_arr[first])
        mean_activity = np.mean([c.mean() for c in all_counts])
    else:
        data_length = 0
        mean_activity = np.nan

    return k_arr, r_k, stderr, data_length, mean_activity, xs, ys


def MR_estimation(
    all_counts, maxslopes=40, scatterpoints=False, eps=1e-12, ar1_fallback=True
):
    """Stable MR estimator (improved numerics)."""
    counts_list = input_handler(all_counts)

    maxslopes = int(max(2, maxslopes))
    max_possible = min(maxslopes, max([len(c) - 1 for c in counts_list]))
    if max_possible < 1:
        raise ValueError("Time series too short for MR estimation")

    k, r_k_raw, stderr_raw, data_length, mean_activity, xs, ys = get_slopes(
        counts_list, max_possible + 1, scatterpoints=scatterpoints, eps=eps
    )

    r_k = np.asarray(r_k_raw, dtype=np.float64)
    stderr = np.asarray(stderr_raw, dtype=np.float64)

    valid = np.isfinite(r_k) & (r_k > 0) & (stderr > 0)
    if np.sum(valid) < 2:
        if ar1_fallback:
            x0 = []
            x1 = []
            for c in counts_list:
                c = np.asarray(c, dtype=np.float64)
                if len(c) > 1:
                    x0.append(c[:-1] - c[:-1].mean())
                    x1.append(c[1:] - c[1:].mean())
            if x0:
                x0 = np.concatenate(x0)
                x1 = np.concatenate(x1)
                var0 = np.mean(x0 * x0)
                cov01 = np.mean(x0 * x1)
                if var0 > eps:
                    br_hat = cov01 / var0
                    return {
                        "branching_ratio": float(br_hat),
                        "autocorrelationtime": float(-1.0 / np.log(br_hat))
                        if br_hat > 0
                        else np.nan,
                        "naive_branching_ratio": float(br_hat),
                        "k": np.array([1]),
                        "r_k": np.array([br_hat]),
                        "stderr": np.array([np.nan]),
                        "data_length": data_length,
                        "mean_activity": mean_activity,
                        "note": "AR1 fallback used (insufficient valid MR points)",
                    }
        raise RuntimeError("Insufficient valid r_k values for MR estimation")

    k_valid = k[valid]
    r_valid = r_k[valid]
    stderr_valid = stderr[valid]

    r_valid = np.clip(r_valid, 1e-12, 1e12)

    y = np.log(r_valid)
    x_design = k_valid.astype(np.float64)

    w = 1.0 / (stderr_valid + eps)
    w = w / np.mean(w)

    p = np.polyfit(x_design, y, 1, w=w)
    logb = p[0]
    loga = p[1]
    b = float(np.exp(logb))
    a = float(np.exp(loga))

    if not np.isfinite(b) or b <= 0:
        raise RuntimeError("Fitted branching ratio b is non-positive/invalid")

    result = {
        "branching_ratio": b,
        "a_fit": a,
        "autocorrelationtime": float(-1.0 / np.log(b))
        if (b > 0 and b != 1.0)
        else np.inf,
        "naive_branching_ratio": float(r_k[0]) if len(r_k) > 0 else np.nan,
        "k": k,
        "r_k": r_k,
        "stderr": stderr,
        "fit_logb": float(logb),
        "fit_loga": float(loga),
        "fit_points_used": int(np.sum(valid)),
        "data_length": data_length,
        "mean_activity": mean_activity,
    }
    if scatterpoints:
        result["xs"] = xs
        result["ys"] = ys

    return result
