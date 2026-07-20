def lambda_key(lam: float) -> str:
    """Stable string key for a lambda value, e.g. 0.5 → '0_5', 1 → '1', 4.0 → '4'."""
    return f"{lam:g}".replace(".", "_").replace("-", "m")


def kernel_sums_exponential_hamming(X: torch.Tensor, Y: torch.Tensor, lambdas,
                                    chunk_size: int = 1024, device=None) -> dict:
    """Compute the sum of the exponential normalized-Hamming kernel over all
    (i, j) pairs in X × Y (including diagonal when X is Y).

    No diagonal exclusion is performed here.  If you want the off-diagonal sum
    (needed for the unbiased U-statistic self-term), subtract the diagonal
    contribution in the caller.

    Returns:
        dict mapping lam → float sum  (shape: scalar per lambda)
    """
    if device is None:
        device = X.device

    m = X.shape[0]
    n = Y.shape[0]
    N_dim = X.shape[1]
    sums = {lam: 0.0 for lam in lambdas}

    X_dev = X.to(device=device, dtype=torch.float32)
    Y_dev = Y.to(device=device, dtype=torch.float32)

    for i_start in range(0, m, chunk_size):
        i_end = min(i_start + chunk_size, m)
        X_chunk = X_dev[i_start:i_end]

        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            Y_chunk = Y_dev[j_start:j_end]

            overlaps = torch.matmul(X_chunk, Y_chunk.t()) / N_dim
            norm_hamming = (1.0 - overlaps) / 2.0

            for lam in lambdas:
                sums[lam] += torch.exp(-lam * norm_hamming).sum().item()

    return sums


def normalized_mmd_weights(lambdas, weights):
    """Return a dict {lam: weight} normalized to sum to 1.

    If weights is None, uniform weights are returned (default uniform mixture).
    """
    if weights is None:
        return {lam: 1.0 / len(lambdas) for lam in lambdas}
    if len(weights) != len(lambdas):
        raise ValueError("mmd_weights must match mmd_lambdas length")
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("mmd_weights must be nonnegative")
    if weights.sum() <= 0:
        raise ValueError("mmd_weights must have positive sum")
    weights = weights / weights.sum()
    return {lam: float(w) for lam, w in zip(lambdas, weights)}


def compute_mmd_biased_unbiased(X: torch.Tensor, Y: torch.Tensor, lambdas,
                                chunk_size: int = 1024, device=None,
                                weights=None) -> dict:
    """Compute biased (V-stat) and unbiased (U-stat) MMD^2 for each lambda,
    plus mixture-kernel summaries.

    The unbiased estimator for the self-term excludes diagonal entries:
        sum_{i != j} k(X_i, X_j) = sum_{i,j} k(X_i, X_j) - sum_i k(X_i, X_i)

    The cross term uses ALL (m×n) pairs — no diagonal exclusion for X≠Y.

    Mixture-kernel MMD² is a weighted sum over lambdas (uniform by default).
    This is valid because a nonneg-weighted sum of PD kernels is PD.

    Returns dict with per-lambda and mixture-kernel entries.
    """
    m = len(X)
    n = len(Y)

    w = normalized_mmd_weights(lambdas, weights)

    # Full sums including diagonal
    sum_xx = kernel_sums_exponential_hamming(X, X, lambdas, chunk_size=chunk_size, device=device)
    sum_yy = kernel_sums_exponential_hamming(Y, Y, lambdas, chunk_size=chunk_size, device=device)
    sum_xy = kernel_sums_exponential_hamming(X, Y, lambdas, chunk_size=chunk_size, device=device)

    # Diagonal of each self-kernel (k(x,x) = exp(0) = 1 for Hamming kernel)
    # Since k_lambda(x,x) = exp(-lambda * 0) = 1 for all lambda.
    diag_xx = float(m)    # sum_i k(X_i, X_i) = m × 1
    diag_yy = float(n)    # sum_i k(Y_i, Y_i) = n × 1

    results = {}
    biased_vals   = []
    unbiased_vals = []
    w_list        = []

    for lam in lambdas:
        key = lambda_key(lam)

        # Biased V-statistic
        mmd2_biased = (sum_xx[lam] / (m * m)
                       + sum_yy[lam] / (n * n)
                       - 2.0 * sum_xy[lam] / (m * n))

        # Unbiased U-statistic (off-diagonal self-terms)
        xx_offdiag = sum_xx[lam] - diag_xx
        yy_offdiag = sum_yy[lam] - diag_yy
        xx_u = xx_offdiag / (m * (m - 1)) if m > 1 else 0.0
        yy_u = yy_offdiag / (n * (n - 1)) if n > 1 else 0.0
        # Cross term: all m*n pairs (no diagonal exclusion for two-sample X≠Y)
        xy   = sum_xy[lam] / (m * n)
        mmd2_unbiased = xx_u + yy_u - 2.0 * xy

        results[f"mmd2_biased_lambda_{key}"]        = mmd2_biased
        results[f"mmd2_unbiased_lambda_{key}_raw"]  = mmd2_unbiased
        results[f"mmd_biased_lambda_{key}"]         = np.sqrt(max(mmd2_biased, 0.0))

        biased_vals.append(mmd2_biased)
        unbiased_vals.append(mmd2_unbiased)
        w_list.append(w[lam])

    # Mixture-kernel summaries (weighted sum; uniform by default)
    w_arr = np.array(w_list)
    mmd2_biased_mix   = float(np.dot(w_arr, biased_vals))
    mmd2_unbiased_mix = float(np.dot(w_arr, unbiased_vals))

    results["mmd2_biased_mixture"]         = mmd2_biased_mix
    results["mmd2_unbiased_mixture_raw"]   = mmd2_unbiased_mix
    results["mmd_biased_mixture"]          = np.sqrt(max(mmd2_biased_mix, 0.0))

    # Backward-compatible multiscale aliases
    results["mmd2_biased_multiscale"]        = results["mmd2_biased_mixture"]
    results["mmd2_unbiased_multiscale_raw"]  = results["mmd2_unbiased_mixture_raw"]
    results["mmd_biased_multiscale"]         = results["mmd_biased_mixture"]

    return results
