import torch
from typing import Optional

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
    "different classes/dg_feature_visualization/dg_features.pt",
    map_location="cpu",
)

dg_patterns = data["dg_features"]
labels = data["class_labels"]
print(dg_patterns[0])
spike_times = dg_to_ttfs(
    dg_patterns,
    t_min_ms=1010,
    t_max_ms=4010,
)

print("DG patterns:", dg_patterns.shape)
print("Spike times:", spike_times.shape)
print("First sample:", spike_times[0])

import os
import torch


def save_ttfs_as_text_files(
    spike_times: torch.Tensor,
    output_dir: str,
    labels: Optional[torch.Tensor] = None,
    no_spike_value: float = -1.0,
    sort_by_time: bool = True,
    include_header: bool = True,
):
    """
    Export TTFS spike times into one text file per sample.

    Args:
        spike_times:
            Tensor shaped [num_samples, num_neurons].

            A valid value represents the neuron's first-spike time.
            `no_spike_value` represents silence.

        output_dir:
            Directory where the text files will be written.

        labels:
            Optional tensor shaped [num_samples].

        no_spike_value:
            Value used to identify neurons that did not spike.

        sort_by_time:
            Sort events chronologically inside each file.

        include_header:
            Add a comment header to each output file.

    File format:
        neuron_id timestamp_ms
    """
    if spike_times.ndim != 2:
        raise ValueError(
            "spike_times must have shape [samples, neurons], "
            f"but received {tuple(spike_times.shape)}."
        )

    spike_times = spike_times.detach().cpu().float()

    if labels is not None:
        labels = labels.detach().cpu()

        if len(labels) != spike_times.size(0):
            raise ValueError(
                "Number of labels does not match number of samples."
            )

    os.makedirs(output_dir, exist_ok=True)

    total_exported_spikes = 0

    for sample_id in range(spike_times.size(0)):
        sample_times = spike_times[sample_id]

        # Identify neurons that produced a spike.
        active_neurons = torch.where(
            sample_times != no_spike_value
        )[0]

        active_times = sample_times[active_neurons]

        if sort_by_time and active_times.numel() > 0:
            order = torch.argsort(active_times)
            active_neurons = active_neurons[order]
            active_times = active_times[order]

        output_path = os.path.join(
            output_dir,
            f"sample_{sample_id:06d}.txt",
        )

        with open(output_path, "w", encoding="utf-8") as file:
            if include_header:
                file.write("# neuron_id timestamp_ms\n")
                file.write(f"# sample_id {sample_id}\n")

                if labels is not None:
                    file.write(
                        f"# class_label {int(labels[sample_id].item())}\n"
                    )

            for neuron_id, timestamp in zip(
                active_neurons.tolist(),
                active_times.tolist(),
            ):
                file.write(
                    f"{neuron_id} {timestamp:.6f}\n"
                )

        total_exported_spikes += len(active_neurons)

    print("=" * 70)
    print("TTFS TEXT EXPORT COMPLETE")
    print("=" * 70)
    print(f"Samples exported:       {spike_times.size(0)}")
    print(f"DG output neurons:      {spike_times.size(1)}")
    print(f"Total spikes exported:  {total_exported_spikes}")
    print(
        "Mean spikes/sample:   "
        f"{total_exported_spikes / spike_times.size(0):.2f}"
    )
    print(f"Output directory:       {output_dir}")
    print("=" * 70)

save_ttfs_as_text_files(
    spike_times=spike_times,
    output_dir="dg_ttfs_text_files",
    labels=labels,
    no_spike_value=-1.0,
)
# import numpy as np
# import torch

# try:
#     from scipy.stats import spearmanr
# except ImportError:
#     spearmanr = None

# def validate_ttfs_conversion(
#     dg_patterns: torch.Tensor,
#     spike_times: torch.Tensor,
#     t_min_ms: float = 1.0,
#     t_max_ms: float = 50.0,
#     activation_threshold: float = 0.0,
#     no_spike_value: float = -1.0,
#     atol: float = 1e-5,
# ):
#     """
#     Validate a time-to-first-spike conversion.

#     Assumptions:
#         dg_patterns: [samples, dg_neurons]
#         spike_times: [samples, dg_neurons]
#         spike_times == no_spike_value means silence
#         positive activations above activation_threshold produce spikes
#     """
#     if dg_patterns.ndim != 2:
#         raise ValueError(
#             f"dg_patterns must be 2D, received {tuple(dg_patterns.shape)}"
#         )

#     if spike_times.ndim != 2:
#         raise ValueError(
#             f"spike_times must be 2D, received {tuple(spike_times.shape)}"
#         )

#     if dg_patterns.shape != spike_times.shape:
#         raise ValueError(
#             "Shape mismatch: "
#             f"DG={tuple(dg_patterns.shape)}, "
#             f"TTFS={tuple(spike_times.shape)}"
#         )

#     dg_patterns = dg_patterns.detach().cpu().float()
#     spike_times = spike_times.detach().cpu().float()

#     # This must match the active-mask rule used by your converter.
#     expected_active = dg_patterns > activation_threshold
#     actual_active = spike_times != no_spike_value

#     expected_counts = expected_active.sum(dim=1)
#     actual_counts = actual_active.sum(dim=1)

#     # ------------------------------------------------------------
#     # 1. Basic tensor validity
#     # ------------------------------------------------------------
#     dg_has_invalid = (
#         torch.isnan(dg_patterns).any()
#         or torch.isinf(dg_patterns).any()
#     )

#     active_spike_times = spike_times[actual_active]

#     spikes_have_invalid = (
#         torch.isnan(active_spike_times).any()
#         or torch.isinf(active_spike_times).any()
#     )

#     # ------------------------------------------------------------
#     # 2. Valid spike-time range
#     # ------------------------------------------------------------
#     if active_spike_times.numel() > 0:
#         valid_time_range = bool(
#             (
#                 (active_spike_times >= t_min_ms - atol)
#                 & (active_spike_times <= t_max_ms + atol)
#             ).all()
#         )
#     else:
#         valid_time_range = False

#     # ------------------------------------------------------------
#     # 3. Active-mask preservation
#     # ------------------------------------------------------------
#     mask_matches = expected_active == actual_active
#     mask_accuracy = mask_matches.float().mean().item()

#     samples_with_exact_mask = (
#         mask_matches.all(dim=1).float().mean().item()
#     )

#     mismatched_entries = (~mask_matches).sum().item()

#     # ------------------------------------------------------------
#     # 4. Spike-count preservation
#     # ------------------------------------------------------------
#     count_matches = expected_counts == actual_counts

#     count_match_fraction = (
#         count_matches.float().mean().item()
#     )

#     mean_expected_count = (
#         expected_counts.float().mean().item()
#     )

#     mean_actual_count = (
#         actual_counts.float().mean().item()
#     )

#     # ------------------------------------------------------------
#     # 5. Monotonicity:
#     # larger activation should produce earlier/equal spike time
#     # ------------------------------------------------------------
#     monotonicity_violations = 0
#     comparisons = 0

#     per_sample_spearman = []

#     for sample_id in range(dg_patterns.size(0)):
#         mask = expected_active[sample_id]

#         activations = dg_patterns[sample_id, mask]
#         times = spike_times[sample_id, mask]

#         if activations.numel() < 2:
#             continue

#         # Pairwise check:
#         # activation_i > activation_j should imply time_i <= time_j
#         activation_difference = (
#             activations.unsqueeze(1)
#             - activations.unsqueeze(0)
#         )

#         time_difference = (
#             times.unsqueeze(1)
#             - times.unsqueeze(0)
#         )

#         strictly_stronger = activation_difference > atol

#         # Violation if a stronger activation spikes later.
#         violations = (
#             strictly_stronger
#             & (time_difference > atol)
#         )

#         monotonicity_violations += violations.sum().item()
#         comparisons += strictly_stronger.sum().item()

#         if spearmanr is not None:
#             activation_np = activations.numpy()
#             time_np = times.numpy()

#             # Expected correlation is negative because high value = early time.
#             correlation = spearmanr(
#                 activation_np,
#                 time_np,
#             ).correlation

#             if not np.isnan(correlation):
#                 per_sample_spearman.append(correlation)

#     violation_fraction = (
#         monotonicity_violations / comparisons
#         if comparisons > 0
#         else None
#     )

#     mean_spearman = (
#         float(np.mean(per_sample_spearman))
#         if per_sample_spearman
#         else None
#     )

#     results = {
#         "dg_shape": tuple(dg_patterns.shape),
#         "spike_shape": tuple(spike_times.shape),
#         "dg_contains_nan_or_inf": bool(dg_has_invalid),
#         "spikes_contain_nan_or_inf": bool(spikes_have_invalid),
#         "valid_time_range": valid_time_range,
#         "mask_accuracy": mask_accuracy,
#         "samples_with_exact_active_mask": samples_with_exact_mask,
#         "mismatched_neuron_entries": int(mismatched_entries),
#         "spike_count_match_fraction": count_match_fraction,
#         "mean_expected_spikes_per_sample": mean_expected_count,
#         "mean_actual_spikes_per_sample": mean_actual_count,
#         "monotonicity_comparisons": int(comparisons),
#         "monotonicity_violations": int(
#             monotonicity_violations
#         ),
#         "monotonicity_violation_fraction": violation_fraction,
#         "mean_activation_time_spearman": mean_spearman,
#     }

#     print("=" * 72)
#     print("TTFS CONVERSION VALIDATION")
#     print("=" * 72)
#     print(f"DG shape:                    {results['dg_shape']}")
#     print(f"TTFS shape:                  {results['spike_shape']}")
#     print(
#         "Valid spike-time range:      ",
#         results["valid_time_range"],
#     )
#     print(
#         "Active-mask accuracy:        ",
#         f"{100 * mask_accuracy:.4f}%",
#     )
#     print(
#         "Samples with exact mask:     ",
#         f"{100 * samples_with_exact_mask:.4f}%",
#     )
#     print(
#         "Spike-count match:           ",
#         f"{100 * count_match_fraction:.4f}%",
#     )
#     print(
#         "Expected spikes/sample:      ",
#         f"{mean_expected_count:.2f}",
#     )
#     print(
#         "Actual spikes/sample:        ",
#         f"{mean_actual_count:.2f}",
#     )

#     if violation_fraction is not None:
#         print(
#             "Monotonicity violations:     ",
#             f"{100 * violation_fraction:.6f}%",
#         )

#     if mean_spearman is not None:
#         print(
#             "Mean activation/time corr.:  ",
#             f"{mean_spearman:.4f}",
#             "(ideally near -1)",
#         )

#     print("=" * 72)

#     passed = (
#         not results["dg_contains_nan_or_inf"]
#         and not results["spikes_contain_nan_or_inf"]
#         and results["valid_time_range"]
#         and results["mask_accuracy"] == 1.0
#         and results["spike_count_match_fraction"] == 1.0
#         and (
#             results["monotonicity_violation_fraction"] is None
#             or results[
#                 "monotonicity_violation_fraction"
#             ] == 0.0
#         )
#     )

#     print(
#         "OVERALL:",
#         "PASS" if passed else "CHECK FAILED TESTS",
#     )

#     results["passed"] = passed

#     return results
# validation = validate_ttfs_conversion(
#     dg_patterns=dg_patterns,
#     spike_times=spike_times,
#     t_min_ms=1010,
#     t_max_ms=4010,
#     activation_threshold=0.0,
# )

# def inspect_ttfs_sample(
#     dg_patterns,
#     spike_times,
#     sample_id=0,
#     threshold=0.0,
# ):
#     pattern = dg_patterns[sample_id]
#     times = spike_times[sample_id]

#     active_indices = torch.where(
#         pattern > threshold
#     )[0]

#     activations = pattern[active_indices]
#     active_times = times[active_indices]

#     # Sort by activation, strongest first.
#     order = torch.argsort(
#         activations,
#         descending=True,
#     )

#     print(
#         f"\nSample {sample_id}: "
#         f"{len(active_indices)} active DG neurons"
#     )

#     print("-" * 55)
#     print(
#         f"{'Neuron':>8} "
#         f"{'Activation':>14} "
#         f"{'Spike time':>14}"
#     )
#     print("-" * 55)

#     for index in order:
#         neuron = int(active_indices[index].item())
#         activation = float(activations[index].item())
#         spike_time = float(active_times[index].item())

#         print(
#             f"{neuron:>8} "
#             f"{activation:>14.6f} "
#             f"{spike_time:>12.3f} ms"
#         )

# inspect_ttfs_sample(
#     dg_patterns,
#     spike_times,
#     sample_id=0,
# )

# import matplotlib.pyplot as plt


# def plot_activation_vs_spike_time(
#     dg_patterns,
#     spike_times,
#     sample_id=0,
#     threshold=0.0,
#     save_path=None,
# ):
#     pattern = dg_patterns[sample_id]
#     times = spike_times[sample_id]

#     active = pattern > threshold

#     activations = pattern[active].numpy()
#     active_times = times[active].numpy()

#     plt.figure(figsize=(7, 5))
#     plt.scatter(
#         activations,
#         active_times,
#         s=35,
#         alpha=0.8,
#     )
#     plt.xlabel("DG activation")
#     plt.ylabel("TTFS spike time (ms)")
#     plt.title(
#         f"DG activation versus spike time — sample {sample_id}"
#     )
#     plt.grid(alpha=0.25)
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(
#             save_path,
#             dpi=200,
#             bbox_inches="tight",
#         )
#         plt.close()
#     else:
#         plt.show()

# plot_activation_vs_spike_time(
#     dg_patterns,
#     spike_times,
#     sample_id=0,
#     threshold=0.0,
#     save_path='verification_features_spike.png',
# )

# def reconstruct_normalized_activation(
#     spike_times,
#     t_min_ms=1.0,
#     t_max_ms=50.0,
#     no_spike_value=-1.0,
# ):
#     active = spike_times != no_spike_value

#     reconstructed = torch.zeros_like(spike_times)

#     reconstructed[active] = (
#         1.0
#         - (
#             spike_times[active] - t_min_ms
#         )
#         / (t_max_ms - t_min_ms)
#     )

#     return reconstructed.clamp(0.0, 1.0)

# positive_patterns = torch.clamp(
#     dg_patterns,
#     min=0.0,
# )

# teacher_normalized = (
#     positive_patterns
#     / positive_patterns.max(
#         dim=1,
#         keepdim=True,
#     ).values.clamp_min(1e-8)
# )

# reconstructed = reconstruct_normalized_activation(
#     spike_times,
#     t_min_ms=1010,
#     t_max_ms=4010,
# )

# active = teacher_normalized > 0

# reconstruction_mae = torch.abs(
#     teacher_normalized[active]
#     - reconstructed[active]
# ).mean()

# print(
#     "Normalized activation reconstruction MAE:",
#     reconstruction_mae.item(),
# )

# teacher_mask = dg_patterns > 0
# spike_mask = spike_times >= 0

# intersection = (
#     teacher_mask & spike_mask
# ).sum(dim=1).float()

# union = (
#     teacher_mask | spike_mask
# ).sum(dim=1).float()

# jaccard = intersection / union.clamp_min(1)

# print(
#     "Mean teacher/TTFS Jaccard:",
#     jaccard.mean().item(),
# )