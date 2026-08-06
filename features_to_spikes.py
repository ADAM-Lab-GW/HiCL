import torch


def dg_to_ttfs(
    dg_patterns: torch.Tensor,
    t_min_ms: float = 1.0,
    t_max_ms: float = 50.0,
    threshold: float = 0.0,
    normalize_per_sample: bool = True,
) -> torch.Tensor:
    """
    Convert sparse DG patterns to time-to-first-spike representations.

    Args:
        dg_patterns:
            Tensor shaped [num_samples, dg_dim].

        t_min_ms:
            Earliest possible spike time.

        t_max_ms:
            Latest possible spike time.

        threshold:
            Activations at or below this value produce no spike.

        normalize_per_sample:
            Normalize each sample independently when True.

    Returns:
        spike_times:
            Tensor shaped [num_samples, dg_dim].

            Active entries contain spike times in milliseconds.
            Inactive entries contain -1.
    """
    if dg_patterns.ndim != 2:
        raise ValueError(
            "Expected DG patterns with shape [samples, dg_dim], "
            f"received {tuple(dg_patterns.shape)}."
        )

    patterns = dg_patterns.float().clone()

    # For a standard excitatory input population, retain positive values.
    patterns = torch.clamp(patterns, min=0.0)

    active_mask = patterns > threshold

    if normalize_per_sample:
        maximum = patterns.max(dim=1, keepdim=True).values
        normalized = patterns / maximum.clamp_min(1e-8)
    else:
        maximum = patterns.max()
        normalized = patterns / maximum.clamp_min(1e-8)

    spike_times = (
        t_min_ms
        + (1.0 - normalized) * (t_max_ms - t_min_ms)
    )

    # -1 means that the neuron does not spike.
    spike_times[~active_mask] = -1.0

    return spike_times

data = torch.load(
    "expert_0_dg_pattern_separation.pt",
    map_location="cpu",
)

dg_patterns = data["separated_patterns"]
labels = data["labels"]

spike_times = dg_to_ttfs(
    dg_patterns,
    t_min_ms=1.0,
    t_max_ms=50.0,
)

print("DG patterns:", dg_patterns.shape)
print("Spike times:", spike_times.shape)
print("First sample:", spike_times[0])