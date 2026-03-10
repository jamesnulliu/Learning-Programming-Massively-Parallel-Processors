import torch


def kl_divergence_from_log_probs(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    reduction: str = "batchmean",
    eps: float = 0.0,
) -> torch.Tensor:
    """
    KL(P || Q) = sum(P * (log P - log Q))
    where log_p = log P, log_q = log Q along the last dim.

    Args:
        log_p: (..., K) log-probabilities for P (target)
        log_q: (..., K) log-probabilities for Q (prediction)
        reduction: "none" | "sum" | "mean" | "batchmean"
        eps: optional additive smoothing in prob space; usually keep 0.0

    Returns:
        KL divergence with the chosen reduction.
    """
    if eps != 0.0:
        # Smooth in prob space then re-normalize
        p = log_p.exp()
        q = log_q.exp()
        p = p + eps
        q = q + eps
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        log_p = (p.clamp_min(1e-30)).log()
        log_q = (q.clamp_min(1e-30)).log()

    p = log_p.exp()
    kl_per_elem = p * (log_p - log_q)  # (..., K)
    kl = kl_per_elem.sum(dim=-1)  # (...,)

    if reduction == "none":
        return kl  # Shape: (...)
    if reduction == "sum":
        return kl.sum()  # Scalar
    if reduction == "mean":
        return kl.mean()  # Scalar
    if reduction == "batchmean":
        return kl.sum() / kl.shape[0]  # Scalar
    raise ValueError(f"Unknown reduction: {reduction}")
