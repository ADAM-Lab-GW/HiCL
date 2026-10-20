#!/usr/bin/env python3
"""
dg_target_codes.py
==================

Target-code specification, dataset construction, supervised training and
metric suite for a topographic sparse DG code on Bars-and-Stripes.

Designed to sit alongside `l_bars_and_stripes_32x32.py` in the HiCL fork.
It mirrors that file's DG operation --

    z      = LayerNorm(ReLU(W_DG x + b_DG))
    p_sep  = TopK(z, k)

-- but adds an explicit *target* code per stimulus so the DG can be trained
supervised, and adds the metrics needed to check whether the learned code
actually has the properties we asked for.


WHAT THE TARGET CODES ARE (AND ARE NOT)
---------------------------------------
The target codes are a SPECIFICATION, not data. They encode a hypothesis
about what a good DG code looks like:

  1. TOPOGRAPHY   - stimuli that are similar in descriptor space get
                    assemblies whose centroids are close on the GC sheet.
  2. UNIQUENESS   - distinct base stimuli get assemblies with low membership
                    overlap, even when their centroids are close.
  3. NOISE TOLERANCE - corrupted versions of the same base stimulus map to the
                    SAME target code. This is where "similar inputs -> similar
                    codes" lives, and it does not conflict with separation
                    across distinct stimuli.
  4. TTFS ORDER   - each active unit gets a first-spike latency, monotone in
                    its drive, so the code carries rank-order information.

Properties 1 and 2 trade off against sparsity. See `feasibility_report()` and
`sigma_for_overlap()`; the trade-off is real and is worth reporting as a
result rather than hiding behind a chosen operating point.

When you later run the unsupervised spiking model, compare it to these targets
on METRICS (overlap, centroid distance, rank correlation) -- never cell-by-cell.
Cell identity here is arbitrary; the geometry is the claim.


USAGE
-----
    # Build codes + inspect feasibility, no training
    python dg_target_codes.py --report-only

    # Sweep the sparsity/sigma frontier
    python dg_target_codes.py --frontier

    # Build codes, train the DG supervised, evaluate vs matched-k baseline
    python dg_target_codes.py --epochs 60 --out-dir ./dg_targets_run

    # Export codes for the spiking model
    python dg_target_codes.py --report-only --export codes.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:  # metrics + code generation still work without torch
    _HAS_TORCH = False
    Dataset = object  # type: ignore


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class DGCodeConfig:
    # --- stimulus ---
    grid_size: int = 32                    # BAS grid (32 -> 1024-D input)
    n_base_patterns: int = 120             # distinct canonical stimuli
    pattern_mode: str = "family"           # 'family' (graded) or 'random'
    n_families: int = 12
    family_spread: int = 3                 # bars flipped from the family seed
    corruption_levels: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20)
    variants_per_level: int = 12
    heldout_corruption_levels: Tuple[float, ...] = (0.15,)  # extrapolation test
    heldout_variants_per_level: int = 12

    # --- descriptor space (defines what "similar" means) ---
    # 'rowcol' : [row means ; col means]. Bars and stripes separate cleanly and
    #            grade within class. Recommended.
    # 'pixels' : raw flattened image. Note that in BAS, pixel similarity and
    #            class membership DISSOCIATE -- a bars/stripes pair can be more
    #            pixel-similar than two same-class patterns.
    descriptor: str = "rowcol"

    # --- GC sheet ---
    sheet_h: int = 64
    sheet_w: int = 64
    toroidal: bool = True                  # avoids edge artifacts in statistics
    sparsity: float = 0.02                 # fraction of GC units active
    sigma: Optional[float] = None          # None -> solved from target_overlap
    target_overlap: float = 0.10           # k / effective_pool_size

    # --- TTFS ---
    t_min: float = 5.0                     # ms
    t_max: float = 40.0                    # ms
    latency_jitter: float = 0.05           # fraction of (t_max - t_min)

    # --- SOM ---
    som_epochs: int = 200
    som_lr0: float = 0.5
    som_sigma0: Optional[float] = None     # None -> sheet_h / 2

    # --- misc ---
    seed: int = 0

    @property
    def dg_dim(self) -> int:
        return self.sheet_h * self.sheet_w

    @property
    def input_dim(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def k(self) -> int:
        return max(1, int(np.ceil(self.sparsity * self.dg_dim)))


# ============================================================================
# SHEET GEOMETRY
# ============================================================================

class Sheet:
    """2D granule-cell sheet with optional toroidal metric."""

    def __init__(self, h: int, w: int, toroidal: bool = True):
        self.h, self.w, self.toroidal = h, w, toroidal
        self.n = h * w
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        self.coords = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float64)

    def _axis_delta(self, a: np.ndarray, b: np.ndarray, length: int) -> np.ndarray:
        d = np.abs(a - b)
        if self.toroidal:
            d = np.minimum(d, length - d)
        return d

    def dist_to_point(self, point: np.ndarray) -> np.ndarray:
        """Distance from every unit to a single (row, col) point."""
        dr = self._axis_delta(self.coords[:, 0], point[0], self.h)
        dc = self._axis_delta(self.coords[:, 1], point[1], self.w)
        return np.sqrt(dr ** 2 + dc ** 2)

    def pairwise_point_dist(self, pts: np.ndarray) -> np.ndarray:
        """Pairwise distances between an (n, 2) array of sheet points."""
        dr = self._axis_delta(pts[:, None, 0], pts[None, :, 0], self.h)
        dc = self._axis_delta(pts[:, None, 1], pts[None, :, 1], self.w)
        return np.sqrt(dr ** 2 + dc ** 2)

    def cross_point_dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """(n,2) x (m,2) -> (n,m) distances under this sheet's metric."""
        dr = self._axis_delta(A[:, None, 0], B[None, :, 0], self.h)
        dc = self._axis_delta(A[:, None, 1], B[None, :, 1], self.w)
        return np.sqrt(dr ** 2 + dc ** 2)

    def centroid(self, unit_idx: np.ndarray) -> np.ndarray:
        """Centroid of a set of units. Circular mean when toroidal."""
        pts = self.coords[unit_idx]
        if not self.toroidal:
            return pts.mean(axis=0)
        out = np.empty(2)
        for ax, length in enumerate((self.h, self.w)):
            ang = 2 * np.pi * pts[:, ax] / length
            m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
            out[ax] = (m % (2 * np.pi)) * length / (2 * np.pi)
        return out


def effective_pool_size(weights: np.ndarray) -> float:
    """Participation ratio 1 / sum(p^2) -- the number of units the Gaussian
    sampling distribution effectively draws from."""
    p = weights / weights.sum()
    return float(1.0 / np.sum(p ** 2))


def sigma_for_overlap(sheet: Sheet, k: int, target_overlap: float,
                      lo: float = 0.5, hi: Optional[float] = None) -> float:
    """Solve for the Gaussian sigma giving k / effective_pool_size ~= target.

    Expected membership overlap between two assemblies drawn from the same
    neighbourhood is ~ k^2 / M_eff, i.e. a fraction k / M_eff of each assembly.
    """
    if hi is None:
        hi = max(sheet.h, sheet.w) / 2.0
    centre = np.array([sheet.h / 2.0, sheet.w / 2.0])
    d = sheet.dist_to_point(centre)

    def ratio(s: float) -> float:
        wts = np.exp(-(d ** 2) / (2 * s ** 2))
        return k / effective_pool_size(wts)

    if ratio(hi) > target_overlap:
        logging.warning(
            "Even sigma=%.1f (whole sheet) gives overlap ratio %.3f > target %.3f. "
            "Lower `sparsity` or enlarge the sheet.", hi, ratio(hi), target_overlap)
        return hi
    for _ in range(60):                     # bisection; ratio decreases in sigma
        mid = 0.5 * (lo + hi)
        if ratio(mid) > target_overlap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================================
# BARS AND STRIPES
# ============================================================================

def make_base_patterns(grid: int, n: int, rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Binary BAS patterns. Returns (images [n, grid, grid], labels [n]).

    Label 0 = bars (rows), 1 = stripes (cols), matching the PennyLane
    convention used in l_bars_and_stripes_32x32.py after remapping.
    """
    seen, imgs, labels = set(), [], []
    attempts = 0
    while len(imgs) < n and attempts < 200 * n:
        attempts += 1
        is_stripes = len(imgs) % 2          # balance the two classes
        mask = rng.random(grid) < 0.5
        if mask.all() or not mask.any():    # all-on / all-off are ambiguous
            continue
        img = np.zeros((grid, grid), dtype=np.float32)
        if is_stripes:
            img[:, mask] = 1.0
        else:
            img[mask, :] = 1.0
        key = (is_stripes, img.tobytes())
        if key in seen:
            continue
        seen.add(key)
        imgs.append(img)
        labels.append(int(is_stripes))
    if len(imgs) < n:
        raise RuntimeError(f"Only generated {len(imgs)}/{n} unique BAS patterns.")
    return np.stack(imgs), np.array(labels, dtype=np.int64)


def make_family_patterns(grid: int, n_families: int, family_size: int,
                         spread: int, rng: np.random.Generator
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """BAS patterns organised into similarity families.

    Random BAS patterns are nearly EQUIDISTANT in descriptor space (see
    `descriptor_concentration`), which leaves topography almost nothing to
    organise. Families fix that: each family shares a seed row/col mask, and
    members differ from it by `spread` flipped bars. Within-family distance is
    small, between-family distance is large.
    """
    seen, imgs, labels = set(), [], []
    for f in range(n_families):
        is_stripes = f % 2
        for _ in range(500):
            seed_mask = rng.random(grid) < 0.5
            if seed_mask.any() and not seed_mask.all():
                break
        members = 0
        for _ in range(500 * family_size):
            if members >= family_size:
                break
            mask = seed_mask.copy()
            n_flip = rng.integers(0, spread + 1)
            if n_flip:
                bits = rng.choice(grid, size=int(n_flip), replace=False)
                mask[bits] = ~mask[bits]
            if mask.all() or not mask.any():
                continue
            img = np.zeros((grid, grid), dtype=np.float32)
            if is_stripes:
                img[:, mask] = 1.0
            else:
                img[mask, :] = 1.0
            key = (is_stripes, img.tobytes())
            if key in seen:
                continue
            seen.add(key)
            imgs.append(img)
            labels.append(int(is_stripes))
            members += 1
    if not imgs:
        raise RuntimeError("Failed to generate family patterns.")
    return np.stack(imgs), np.array(labels, dtype=np.int64)


def descriptor_concentration(desc: np.ndarray) -> Dict[str, float]:
    """Coefficient of variation of pairwise descriptor distances.

    CV below ~0.15 means your stimuli are effectively equidistant and there is
    no graded similarity for a topographic code to express. Check this BEFORE
    interpreting any topography score.
    """
    D = np.linalg.norm(desc[:, None, :] - desc[None, :, :], axis=2)
    v = D[np.triu_indices(len(desc), k=1)]
    return {"dist_mean": float(v.mean()), "dist_sd": float(v.std()),
            "dist_cv": float(v.std() / (v.mean() + 1e-12)),
            "verdict": ("near-equidistant: topography has little to express"
                        if v.std() / (v.mean() + 1e-12) < 0.15
                        else "graded structure present")}


def corrupt(img: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Flip a fraction p of pixels. p=0 returns a copy."""
    if p <= 0:
        return img.copy()
    out = img.copy()
    n_flip = int(round(p * out.size))
    if n_flip == 0:
        return out
    idx = rng.choice(out.size, size=n_flip, replace=False)
    flat = out.ravel()
    flat[idx] = 1.0 - flat[idx]
    return out


def descriptors(imgs: np.ndarray, kind: str) -> np.ndarray:
    """Map images to the space in which 'similarity' is defined."""
    if kind == "pixels":
        return imgs.reshape(len(imgs), -1).astype(np.float64)
    if kind == "rowcol":
        return np.concatenate(
            [imgs.mean(axis=2), imgs.mean(axis=1)], axis=1).astype(np.float64)
    raise ValueError(f"Unknown descriptor kind: {kind}")


def from_pennylane(inputs, labels, grid_size: int = 32
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Adapter for the PennyLane BAS arrays used in the fork.

    Accepts [N, D] / [N, H, W], labels in {-1, +1}, returns float images in
    [0, 1] and labels in {0, 1} (0 = bars).
    """
    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim == 2:
        side = int(round(np.sqrt(x.shape[1])))
        x = x.reshape(-1, side, side)
    if x.shape[-1] != grid_size:
        raise ValueError(f"Expected grid {grid_size}, got {x.shape[-1]}. "
                         "Resize before calling.")
    lo, hi = x.min(), x.max()
    if hi > lo:
        x = (x - lo) / (hi - lo)
    y = (np.asarray(labels, dtype=np.float32).ravel() > 0).astype(np.int64)
    return x, y


# ============================================================================
# SELF-ORGANISING MAP  (places assemblies on the sheet from input geometry)
# ============================================================================

def train_som(X: np.ndarray, sheet: Sheet, epochs: int, lr0: float,
              sigma0: Optional[float], rng: np.random.Generator) -> np.ndarray:
    """Kohonen SOM whose unit lattice *is* the GC sheet. Returns weights."""
    n, d = X.shape
    if sigma0 is None:
        sigma0 = max(sheet.h, sheet.w) / 2.0
    W = X[rng.choice(n, size=sheet.n, replace=True)].copy()
    W += 0.01 * rng.standard_normal(W.shape)
    sigma_end, lr_end = 0.8, 0.01

    for ep in range(epochs):
        frac = ep / max(1, epochs - 1)
        sigma = sigma0 * (sigma_end / sigma0) ** frac
        lr = lr0 * (lr_end / lr0) ** frac
        for i in rng.permutation(n):
            x = X[i]
            bmu = int(np.argmin(((W - x) ** 2).sum(axis=1)))
            dist = sheet.dist_to_point(sheet.coords[bmu])
            h = np.exp(-(dist ** 2) / (2 * sigma ** 2))
            W += (lr * h)[:, None] * (x - W)
    return W


def som_bmu_coords(W: np.ndarray, X: np.ndarray, sheet: Sheet) -> np.ndarray:
    """Best-matching-unit sheet coordinate for each descriptor."""
    d2 = ((X[:, None, :] - W[None, :, :]) ** 2).sum(axis=2)
    return sheet.coords[np.argmin(d2, axis=1)]


# ============================================================================
# TARGET CODE CONSTRUCTION
# ============================================================================

def stable_seed(arr: np.ndarray, salt: int = 0) -> int:
    """Deterministic seed from array contents (unlike Python's hash())."""
    h = hashlib.blake2b(np.ascontiguousarray(arr).tobytes()
                        + salt.to_bytes(8, "little"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def build_target_codes(base_imgs: np.ndarray, centers: np.ndarray, sheet: Sheet,
                       k: int, sigma: float, cfg: DGCodeConfig
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each base stimulus a sparse assembly + TTFS latencies.

    Returns
    -------
    active : (n_base, k) int  -- unit indices, sorted by latency
    latency: (n_base, k) float -- first-spike times in ms, ascending
    """
    n = len(base_imgs)
    active = np.zeros((n, k), dtype=np.int64)
    latency = np.zeros((n, k), dtype=np.float64)
    span = cfg.t_max - cfg.t_min

    for i in range(n):
        rng = np.random.default_rng(stable_seed(base_imgs[i], cfg.seed))
        d = sheet.dist_to_point(centers[i])
        w = np.exp(-(d ** 2) / (2 * sigma ** 2))
        p = w / w.sum()
        idx = rng.choice(sheet.n, size=k, replace=False, p=p)

        # latency monotone in drive: near the centre fires first, plus jitter
        drive = d[idx] + cfg.latency_jitter * sigma * rng.standard_normal(k)
        order = np.argsort(drive)
        idx = idx[order]
        ranks = np.arange(k) / max(1, k - 1)
        active[i] = idx
        latency[i] = cfg.t_min + span * ranks
    return active, latency


def codes_to_dense(active: np.ndarray, latency: np.ndarray, dg_dim: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """(n, k) index/latency form -> dense (n, dg_dim) mask and latency arrays."""
    n = len(active)
    mask = np.zeros((n, dg_dim), dtype=np.float32)
    lat = np.zeros((n, dg_dim), dtype=np.float32)
    rows = np.repeat(np.arange(n), active.shape[1])
    mask[rows, active.ravel()] = 1.0
    lat[rows, active.ravel()] = latency.ravel()
    return mask, lat


# ============================================================================
# METRICS
# ============================================================================

def jaccard_matrix(mask: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard over binary rows. mask: (n, d) in {0,1}."""
    m = mask.astype(np.float64)
    inter = m @ m.T
    sizes = m.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.where(union > 0, inter / union, 0.0)
    return j


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def latency_rank_correlation(active_a, lat_a, active_b, lat_b) -> Optional[float]:
    """Spearman correlation of latencies over units active in BOTH codes.

    Returns None when fewer than 3 units are shared -- rank correlation is
    undefined there, and averaging over NaNs silently biases the result.
    """
    shared = np.intersect1d(active_a, active_b)
    if len(shared) < 3:
        return None
    ia = {u: t for u, t in zip(active_a, lat_a)}
    ib = {u: t for u, t in zip(active_b, lat_b)}
    ra = np.argsort(np.argsort([ia[u] for u in shared]))
    rb = np.argsort(np.argsort([ib[u] for u in shared]))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def topography_score(desc: np.ndarray, centroids: np.ndarray, sheet: Sheet
                     ) -> float:
    """Spearman correlation between descriptor distance and sheet-centroid
    distance over all pairs. High = topographic."""
    dd = np.linalg.norm(desc[:, None, :] - desc[None, :, :], axis=2)
    sd = sheet.pairwise_point_dist(centroids)
    iu = np.triu_indices(len(desc), k=1)
    a, b = dd[iu], sd[iu]
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def code_report(mask: np.ndarray, active: np.ndarray, latency: np.ndarray,
                desc: np.ndarray, sheet: Sheet, label: str = "") -> Dict:
    """Full geometry report for a set of codes."""
    n, d = mask.shape
    k = int(mask.sum(axis=1).mean())
    centroids = np.stack([sheet.centroid(a) for a in active])

    J = jaccard_matrix(mask)
    iu = np.triu_indices(n, k=1)
    j_off = J[iu]

    usage = mask.sum(axis=0)
    dead = int((usage == 0).sum())
    hub = int((usage > 5 * n * k / d).sum())   # 5x expected usage

    rcs = []
    for i in range(min(n, 60)):
        for jx in range(i + 1, min(n, 60)):
            r = latency_rank_correlation(active[i], latency[i],
                                         active[jx], latency[jx])
            if r is not None:
                rcs.append(r)

    return {
        "label": label,
        "n_codes": int(n),
        "dg_dim": int(d),
        "k": k,
        "active_fraction": float(k / d),
        "jaccard_mean": float(j_off.mean()),
        "jaccard_p95": float(np.percentile(j_off, 95)),
        "jaccard_max": float(j_off.max()),
        "topography_spearman": topography_score(desc, centroids, sheet),
        "dead_units": dead,
        "dead_fraction": float(dead / d),
        "hub_units": hub,
        "latency_rank_corr_mean": float(np.mean(rcs)) if rcs else None,
        "n_pairs_sharing_3plus_units": len(rcs),
    }


def centroid_spacing(bc: "BuiltCodes") -> Dict[str, float]:
    """Packing of the target assemblies on the sheet.

    Identification accuracy is governed by the ratio

        encoder centroid error / mean nearest-neighbour centroid spacing

    not by centroid error alone. An error of 3 sheet units is excellent when
    codes are 10 units apart and near-useless when they are 3.9 apart.
    """
    C = np.stack([bc.sheet.centroid(a) for a in bc.active])
    D = bc.sheet.cross_point_dist(C, C)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    return {"nn_spacing_mean": float(nn.mean()),
            "nn_spacing_median": float(np.median(nn)),
            "nn_spacing_p10": float(np.percentile(nn, 10))}


def separation_curve(in_sim: np.ndarray, out_sim: np.ndarray, n_bins: int = 12
                     ) -> Dict[str, List[float]]:
    """Bin pairs by input similarity, report mean output similarity.

    This is the R_in vs R_out curve. Points below the diagonal = separation.
    """
    iu = np.triu_indices(in_sim.shape[0], k=1)
    x, y = in_sim[iu], out_sim[iu]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 5:
            continue
        xs.append(float(x[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return {"r_in": xs, "r_out": ys, "n_pairs": ns}


# ============================================================================
# MATCHED-k RANDOM BASELINE  (the control that matters most)
# ============================================================================

def matched_k_baseline(X: np.ndarray, dg_dim: int, k: int, t_min: float,
                       t_max: float, seed: int = 0
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random projection -> ReLU -> TopK at the SAME k.

    Returns (mask, active_idx, latency) with latencies derived from drive rank,
    the same way the target codes derive them, so the two are comparable.

    Any separation advantage your spiking DG shows must be measured against
    this, not against an unmatched-sparsity model. Sparsity alone decorrelates.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((X.shape[1], dg_dim)) / np.sqrt(X.shape[1])
    z = np.maximum(X @ W, 0.0)
    idx = np.argpartition(-z, kth=k - 1, axis=1)[:, :k]
    # order each row's winners by descending drive -> ascending latency
    order = np.argsort(-np.take_along_axis(z, idx, axis=1), axis=1)
    active = np.take_along_axis(idx, order, axis=1)
    mask = np.zeros_like(z, dtype=np.float32)
    np.put_along_axis(mask, active, 1.0, axis=1)
    ranks = np.arange(k) / max(1, k - 1)
    latency = np.tile(t_min + (t_max - t_min) * ranks, (len(X), 1))
    return mask, active, latency


# ============================================================================
# DATASET ASSEMBLY
# ============================================================================

@dataclass
class BuiltCodes:
    cfg: DGCodeConfig
    sheet: Sheet
    sigma: float
    base_imgs: np.ndarray          # (n_base, g, g)
    base_labels: np.ndarray        # (n_base,)
    base_desc: np.ndarray          # (n_base, d_desc)
    centers: np.ndarray            # (n_base, 2)
    active: np.ndarray             # (n_base, k)
    latency: np.ndarray            # (n_base, k)
    mask: np.ndarray               # (n_base, dg_dim)
    dense_latency: np.ndarray      # (n_base, dg_dim)
    som_weights: np.ndarray = field(default=None, repr=False)


def build_codes(cfg: DGCodeConfig) -> BuiltCodes:
    rng = np.random.default_rng(cfg.seed)
    sheet = Sheet(cfg.sheet_h, cfg.sheet_w, cfg.toroidal)
    k = cfg.k

    sigma = cfg.sigma if cfg.sigma is not None else \
        sigma_for_overlap(sheet, k, cfg.target_overlap)

    if cfg.pattern_mode == "family":
        per = max(2, int(round(cfg.n_base_patterns / cfg.n_families)))
        base_imgs, base_labels = make_family_patterns(
            cfg.grid_size, cfg.n_families, per, cfg.family_spread, rng)
    else:
        base_imgs, base_labels = make_base_patterns(
            cfg.grid_size, cfg.n_base_patterns, rng)
    desc = descriptors(base_imgs, cfg.descriptor)
    desc_n = (desc - desc.mean(0)) / (desc.std(0) + 1e-8)

    W = train_som(desc_n, sheet, cfg.som_epochs, cfg.som_lr0, cfg.som_sigma0, rng)
    centers = som_bmu_coords(W, desc_n, sheet)

    active, latency = build_target_codes(base_imgs, centers, sheet, k, sigma, cfg)
    mask, dense_lat = codes_to_dense(active, latency, cfg.dg_dim)

    return BuiltCodes(cfg=cfg, sheet=sheet, sigma=sigma, base_imgs=base_imgs,
                      base_labels=base_labels, base_desc=desc_n, centers=centers,
                      active=active, latency=latency, mask=mask,
                      dense_latency=dense_lat, som_weights=W)


def make_variants(bc: BuiltCodes, levels: Tuple[float, ...], per_level: int,
                  seed_salt: int, dedupe_clean: bool = True
                  ) -> Dict[str, np.ndarray]:
    """Corrupted variants. Each inherits its BASE pattern's target code."""
    cfg = bc.cfg
    rng = np.random.default_rng(cfg.seed + seed_salt)
    X, base_id, corr, lab = [], [], [], []
    for i, img in enumerate(bc.base_imgs):
        for p in levels:
            reps = 1 if (p == 0.0 and dedupe_clean) else per_level
            for _ in range(reps):
                X.append(corrupt(img, p, rng).ravel())
                base_id.append(i)
                corr.append(p)
                lab.append(bc.base_labels[i])
    return {
        "x": np.stack(X).astype(np.float32),
        "base_id": np.array(base_id, dtype=np.int64),
        "corruption": np.array(corr, dtype=np.float32),
        "label": np.array(lab, dtype=np.int64),
    }


if _HAS_TORCH:
    class DGTargetDataset(Dataset):
        """(input, target mask, target latency, base_id, corruption, label)."""

        def __init__(self, variants: Dict[str, np.ndarray], bc: BuiltCodes):
            self.x = torch.from_numpy(variants["x"])
            self.base_id = torch.from_numpy(variants["base_id"])
            self.corruption = torch.from_numpy(variants["corruption"])
            self.label = torch.from_numpy(variants["label"])
            self.mask = torch.from_numpy(bc.mask)
            self.lat = torch.from_numpy(bc.dense_latency)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            b = self.base_id[i]
            return (self.x[i], self.mask[b], self.lat[b],
                    b, self.corruption[i], self.label[i])


# ============================================================================
# SUPERVISED DG
# ============================================================================

if _HAS_TORCH:
    class SupervisedDG(nn.Module):
        """Mirrors CustomDentateGyrusExpert, with membership + latency heads.

        Latency is tied to drive through a monotone map rather than a free
        head: a cell with more drive fires earlier, as an integrate-and-fire
        cell would. Fewer parameters and a coherent story.
        """

        def __init__(self, input_dim: int, dg_dim: int, k: int,
                     t_min: float, t_max: float):
            super().__init__()
            self.dg_projection = nn.Linear(input_dim, dg_dim)
            self.dg_layer_norm = nn.LayerNorm(dg_dim)
            self.k, self.t_min, self.t_max = k, t_min, t_max
            # per-unit affine terms = heterogeneous excitability / threshold
            self.member_scale = nn.Parameter(torch.ones(dg_dim))
            self.member_bias = nn.Parameter(torch.zeros(dg_dim))
            self.lat_scale_raw = nn.Parameter(torch.zeros(dg_dim))  # softplus -> >0
            self.lat_bias = nn.Parameter(torch.zeros(dg_dim))

        def drive(self, x):
            return self.dg_layer_norm(F.relu(self.dg_projection(x)))

        def forward(self, x):
            z = self.drive(x)
            logits = self.member_scale * z + self.member_bias
            # more drive -> earlier spike; softplus keeps the map monotone
            slope = F.softplus(self.lat_scale_raw) + 1e-3
            frac = torch.sigmoid(-(slope * z + self.lat_bias))
            lat = self.t_min + (self.t_max - self.t_min) * frac
            return logits, lat, z

        @torch.no_grad()
        def topk_mask(self, x):
            z = self.drive(x)
            idx = z.topk(self.k, dim=1).indices
            m = torch.zeros_like(z)
            m.scatter_(1, idx, 1.0)
            return m


    def dg_losses(logits, lat_pred, target_mask, target_lat, k: int,
                  t_min: float, t_max: float, latency_weight: float = 1.0):
        """Weighted BCE on membership + censored latency regression.

        Cells that should have fired and did not are CENSORED, not late --
        the latency term is applied only where the target is active.
        """
        n_units = target_mask.shape[1]
        pos_weight = torch.tensor((n_units - k) / k, device=logits.device)
        l_member = F.binary_cross_entropy_with_logits(
            logits, target_mask, pos_weight=pos_weight)

        span = max(1e-6, t_max - t_min)   # normalise ms -> [0,1] or BCE is swamped
        m = target_mask > 0
        if m.any():
            diff = (lat_pred[m] - target_lat[m]) / span
            l_lat = (diff ** 2).mean()
        else:
            l_lat = torch.zeros((), device=logits.device)
        return l_member + latency_weight * l_lat, l_member.detach(), l_lat.detach()


    def train_supervised(model, train_ds, val_ds, epochs: int, batch_size: int,
                         lr: float, device: str, latency_weight: float,
                         log_every: int = 5, restore_best: bool = True
                         ) -> Tuple[List[Dict], Dict]:
        model.to(device)
        tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        best = {"recall": -1.0, "epoch": 0, "state": None}

        for ep in range(1, epochs + 1):
            model.train()
            agg = np.zeros(3)
            nb = 0
            for x, tm, tlat, _, _, _ in tl:
                x, tm, tlat = x.to(device), tm.to(device), tlat.to(device)
                logits, lat, _ = model(x)
                loss, lm, ll = dg_losses(logits, lat, tm, tlat, model.k,
                                         model.t_min, model.t_max,
                                         latency_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                agg += [loss.item(), lm.item(), ll.item()]
                nb += 1
            agg /= max(1, nb)

            model.eval()
            hits, tot = 0, 0
            with torch.no_grad():
                for x, tm, _, _, _, _ in vl:
                    x, tm = x.to(device), tm.to(device)
                    pm = model.topk_mask(x)
                    hits += (pm * tm).sum().item()
                    tot += tm.sum().item()
            recall = hits / max(1.0, tot)
            history.append({"epoch": ep, "loss": agg[0], "l_member": agg[1],
                            "l_latency": agg[2], "val_topk_recall": recall})
            if recall > best["recall"]:
                best = {"recall": recall, "epoch": ep,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
            if ep % log_every == 0 or ep == 1:
                logging.info("epoch %3d | loss %.4f | member %.4f | lat %.4f "
                             "| val top-k recall %.3f",
                             ep, agg[0], agg[1], agg[2], recall)

        if restore_best and best["state"] is not None:
            model.load_state_dict(best["state"])
            logging.info("\nRestored best checkpoint: epoch %d, val recall %.3f "
                         "(final epoch was %.3f). Report the peak, not the last "
                         "epoch -- training loss keeps falling after validation "
                         "stops improving.", best["epoch"], best["recall"],
                         history[-1]["val_topk_recall"])
        return history, {"best_epoch": best["epoch"], "best_recall": best["recall"],
                         "final_recall": history[-1]["val_topk_recall"]}


    @torch.no_grad()
    def learned_codes(model, x: np.ndarray, device: str, batch: int = 256
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mask, latency) the trained DG assigns to each input."""
        model.eval()
        masks, lats = [], []
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            m = model.topk_mask(xb)
            _, lat, _ = model(xb)
            masks.append(m.cpu().numpy())
            lats.append((lat * m).cpu().numpy())
        return np.concatenate(masks), np.concatenate(lats)


    def diagnostics(model, bc: BuiltCodes, device: str,
                    levels: Tuple[float, ...], per_level: int = 12,
                    seed_salt: int = 77) -> Tuple[List[Dict], Dict]:
        """Three diagnostics that separate 'noise-robustness failure' from
        'capacity failure', and give credit for landing in the right
        neighbourhood with the wrong cells.

          recall            - fraction of the target assembly recovered.
                              Harsh: zero credit for near misses.
          centroid_error    - sheet distance between predicted and target
                              assembly centroid. Survives when recall does not.
          identification    - is the predicted centroid closer to this
                              stimulus's target centroid than to any other?
                              This is the number that matters for routing.
        """
        sheet, cfg = bc.sheet, bc.cfg
        tgt_centroids = np.stack([sheet.centroid(a) for a in bc.active])

        rng = np.random.default_rng(cfg.seed + 999)
        rand_pts = np.stack([sheet.coords[rng.integers(0, sheet.n)]
                             for _ in range(2000)])
        chance_centroid = float(sheet.cross_point_dist(
            rand_pts[:1000], rand_pts[1000:]).diagonal().mean())

        rows = []
        for p in levels:
            v = make_variants(bc, (p,), per_level, seed_salt, dedupe_clean=True)
            mask, _ = learned_codes(model, v["x"], device)
            tgt = bc.mask[v["base_id"]]

            recall = float((mask * tgt).sum(axis=1).mean() / cfg.k)
            pred_c = np.stack([sheet.centroid(np.flatnonzero(r)) for r in mask])
            own = sheet.cross_point_dist(pred_c, tgt_centroids)
            cent_err = float(own[np.arange(len(pred_c)), v["base_id"]].mean())
            ident = float((np.argmin(own, axis=1) == v["base_id"]).mean())

            rows.append({
                "corruption": p,
                "n": len(v["x"]),
                "recall": round(recall, 4),
                "centroid_err": round(cent_err, 3),
                "identification": round(ident, 4),
            })

        chance = {
            "recall_chance": round(cfg.k / cfg.dg_dim, 4),
            "centroid_err_chance": round(chance_centroid, 3),
            "identification_chance": round(1.0 / len(bc.base_imgs), 4),
        }
        return rows, chance


    def capacity_sweep(base_cfg: DGCodeConfig, n_bases: Tuple[int, ...],
                       epochs: int, batch_size: int, lr: float, device: str,
                       latency_weight: float, corruption: float = 0.0
                       ) -> List[Dict]:
        """How does the code degrade as the stimulus count grows?

        For each n_base: rebuild codes, retrain, and measure recall, centroid
        error, identification, and -- crucially -- the centroid packing spacing
        that explains them. This is the capacity curve.
        """
        rows = []
        for n in n_bases:
            logging.info("\n=== capacity sweep: n_base = %d ===", n)
            cfg = DGCodeConfig(**{**asdict(base_cfg), "n_base_patterns": n,
                                  "n_families": max(2, int(round(n / 10)))})
            bc = build_codes(cfg)
            spacing = centroid_spacing(bc)

            train_v = make_variants(bc, cfg.corruption_levels,
                                    cfg.variants_per_level, 1)
            test_v = make_variants(bc, cfg.heldout_corruption_levels,
                                   cfg.heldout_variants_per_level, 2)
            model = SupervisedDG(cfg.input_dim, cfg.dg_dim, cfg.k,
                                 cfg.t_min, cfg.t_max)
            _, best = train_supervised(model, DGTargetDataset(train_v, bc),
                                       DGTargetDataset(test_v, bc), epochs,
                                       batch_size, lr, device, latency_weight,
                                       log_every=max(1, epochs // 2))

            diag, chance = diagnostics(model, bc, device, (corruption,),
                                       per_level=cfg.variants_per_level)
            d = diag[0]
            err = d["centroid_err"]
            rows.append({
                "n_base": n,
                "k": cfg.k,
                "sigma": round(bc.sigma, 2),
                "nn_spacing": round(spacing["nn_spacing_mean"], 2),
                "centroid_err": err,
                "err_over_spacing": round(err / spacing["nn_spacing_mean"], 3),
                "recall": d["recall"],
                "identification": d["identification"],
                "ident_chance": chance["identification_chance"],
                "best_epoch": best["best_epoch"],
            })
        return rows


# ============================================================================
# REPORTING
# ============================================================================

def feasibility_report(cfg: DGCodeConfig) -> Dict:
    """Can a code with these properties exist at this sparsity? Print first,
    generate second. This is the trade-off worth showing your advisor."""
    sheet = Sheet(cfg.sheet_h, cfg.sheet_w, cfg.toroidal)
    k = cfg.k
    sigma = cfg.sigma if cfg.sigma is not None else \
        sigma_for_overlap(sheet, k, cfg.target_overlap)
    centre = np.array([sheet.h / 2.0, sheet.w / 2.0])
    w = np.exp(-(sheet.dist_to_point(centre) ** 2) / (2 * sigma ** 2))
    m_eff = effective_pool_size(w)

    return {
        "dg_dim": cfg.dg_dim,
        "k_active": k,
        "sparsity": cfg.sparsity,
        "sigma": round(sigma, 3),
        "effective_pool_size": round(m_eff, 1),
        "expected_overlap_ratio": round(k / m_eff, 4),
        "expected_jaccard_same_neighbourhood": round(
            (k / m_eff) / (2 - k / m_eff), 4),
        "n_quasi_independent_neighbourhoods": round(cfg.dg_dim / m_eff, 1),
        "n_base_patterns": cfg.n_base_patterns,
        "note": ("Quasi-independent neighbourhoods ~= 1/(10*sparsity) when "
                 "target_overlap=0.1, INDEPENDENT of dg_dim. Enlarging the "
                 "sheet does not buy more separable neighbourhoods -- only "
                 "lowering sparsity does. Centroid position is still resolvable "
                 "far more finely than pool separation, so graded topography "
                 "survives well past this count."),
    }


def frontier_sweep(cfg: DGCodeConfig,
                   sparsities=(0.005, 0.01, 0.02, 0.03, 0.05),
                   overlaps=(0.05, 0.10, 0.20)) -> List[Dict]:
    rows = []
    for s in sparsities:
        for o in overlaps:
            c = DGCodeConfig(**{**asdict(cfg), "sparsity": s,
                                "target_overlap": o, "sigma": None})
            r = feasibility_report(c)
            rows.append({"sparsity": s, "target_overlap": o,
                         "k": r["k_active"], "sigma": r["sigma"],
                         "M_eff": r["effective_pool_size"],
                         "n_neighbourhoods":
                             r["n_quasi_independent_neighbourhoods"]})
    return rows


def print_table(rows: List[Dict], title: str) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = [max(len(c), max(len(f"{r[c]}") for r in rows)) for c in cols]
    print(f"\n{title}")
    print("  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(f"{r[c]}".ljust(w) for c, w in zip(cols, widths)))


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Topographic sparse DG target codes for Bars-and-Stripes.")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-base", type=int, default=120)
    ap.add_argument("--pattern-mode", choices=["family", "random"],
                    default="family")
    ap.add_argument("--n-families", type=int, default=12)
    ap.add_argument("--family-spread", type=int, default=3)
    ap.add_argument("--sheet", type=int, nargs=2, default=[64, 64])
    ap.add_argument("--sparsity", type=float, default=0.02)
    ap.add_argument("--target-overlap", type=float, default=0.10)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--descriptor", choices=["rowcol", "pixels"],
                    default="rowcol")
    ap.add_argument("--som-epochs", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=0,
                    help="supervised epochs; 0 = build codes only")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latency-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capacity-sweep", type=int, nargs="+", default=None,
                    metavar="N",
                    help="rebuild + retrain at each n_base and emit the "
                         "capacity curve, e.g. --capacity-sweep 30 60 90 120 180")
    ap.add_argument("--sweep-epochs", type=int, default=20,
                    help="epochs per sweep point (validation peaks early)")
    ap.add_argument("--sweep-corruption", type=float, default=0.0,
                    help="corruption level at which sweep points are measured")
    ap.add_argument("--diagnostics", action="store_true",
                    help="recall / centroid error / identification by "
                         "corruption level, with chance baselines")
    ap.add_argument("--no-restore-best", action="store_true",
                    help="report the final epoch instead of the validation peak")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--export", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="./dg_targets_run")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = DGCodeConfig(
        grid_size=args.grid_size, n_base_patterns=args.n_base,
        pattern_mode=args.pattern_mode, n_families=args.n_families,
        family_spread=args.family_spread,
        sheet_h=args.sheet[0], sheet_w=args.sheet[1],
        sparsity=args.sparsity, target_overlap=args.target_overlap,
        sigma=args.sigma, descriptor=args.descriptor,
        som_epochs=args.som_epochs, seed=args.seed,
    )

    feas = feasibility_report(cfg)
    print_table([{k: v for k, v in feas.items() if k != "note"}], "FEASIBILITY")
    print("\n  " + feas["note"])

    if args.frontier:
        print_table(frontier_sweep(cfg), "SPARSITY / OVERLAP FRONTIER")
        return

    if args.capacity_sweep:
        if not _HAS_TORCH:
            logging.error("torch not available; cannot run the capacity sweep.")
            return
        dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        rows = capacity_sweep(cfg, tuple(args.capacity_sweep),
                              args.sweep_epochs, args.batch_size, args.lr,
                              dev, args.latency_weight, args.sweep_corruption)
        print_table(rows, "CAPACITY CURVE  (measured at %.0f%% corruption)"
                    % (100 * args.sweep_corruption))
        print("\n  err_over_spacing is the controlling quantity: identification")
        print("  collapses as encoder centroid error approaches the distance")
        print("  between neighbouring target codes. Below ~0.4 identification")
        print("  should be high; near 1.0 you are landing on your neighbour.")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "capacity_curve.json"), "w") as f:
            json.dump({"config": asdict(cfg), "sweep_epochs": args.sweep_epochs,
                       "corruption": args.sweep_corruption, "rows": rows},
                      f, indent=2)
        logging.info("\nCapacity curve -> %s/capacity_curve.json", args.out_dir)
        return

    logging.info("\nBuilding target codes ...")
    bc = build_codes(cfg)
    logging.info("sigma = %.2f, k = %d, dg_dim = %d", bc.sigma, cfg.k, cfg.dg_dim)

    spacing = centroid_spacing(bc)
    print_table([{k: round(v, 3) for k, v in spacing.items()}],
                "TARGET CENTROID PACKING")

    conc = descriptor_concentration(bc.base_desc)
    print_table([{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in conc.items()}], "STIMULUS SIMILARITY STRUCTURE")

    rep = code_report(bc.mask, bc.active, bc.latency, bc.base_desc, bc.sheet,
                      "target codes")
    print_table([{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in rep.items()}], "TARGET CODE GEOMETRY")

    # input-side similarity for the separation curve
    flat = bc.base_imgs.reshape(len(bc.base_imgs), -1)
    in_j = jaccard_matrix((flat > 0.5).astype(np.float32))
    tgt_curve = separation_curve(in_j, jaccard_matrix(bc.mask))

    base_mask, base_active, base_lat = matched_k_baseline(
        flat, cfg.dg_dim, cfg.k, cfg.t_min, cfg.t_max, cfg.seed)
    base_rep = code_report(base_mask, base_active, base_lat,
                           bc.base_desc, bc.sheet, "matched-k random")
    print_table([{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in base_rep.items()}],
                "MATCHED-k RANDOM BASELINE (the control)")

    base_curve = separation_curve(in_j, jaccard_matrix(base_mask))
    print("\nSEPARATION CURVE  (R_in -> R_out; below diagonal = separation)")
    print("  R_in    target   matched-k")
    for i, x in enumerate(tgt_curve["r_in"]):
        b = base_curve["r_out"][i] if i < len(base_curve["r_out"]) else float("nan")
        print(f"  {x:.3f}   {tgt_curve['r_out'][i]:.4f}   {b:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.export:
        np.savez_compressed(
            args.export, base_imgs=bc.base_imgs, base_labels=bc.base_labels,
            active=bc.active, latency=bc.latency, mask=bc.mask,
            centers=bc.centers, sheet_shape=np.array([cfg.sheet_h, cfg.sheet_w]),
            sigma=bc.sigma, config=json.dumps(asdict(cfg)))
        logging.info("\nExported codes -> %s", args.export)

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({"config": asdict(cfg),
                   "feasibility": feas,
                   "concentration": conc, "spacing": spacing,
                   "target": rep, "matched_k": base_rep,
                   "curve_target": tgt_curve, "curve_matched_k": base_curve},
                  f, indent=2)

    if args.report_only or args.epochs == 0:
        logging.info("\nReport written to %s/report.json", args.out_dir)
        return

    if not _HAS_TORCH:
        logging.error("torch not available; cannot train.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_v = make_variants(bc, cfg.corruption_levels, cfg.variants_per_level, 1)
    test_v = make_variants(bc, cfg.heldout_corruption_levels,
                           cfg.heldout_variants_per_level, 2)
    logging.info("\ntrain variants: %d   held-out-corruption variants: %d",
                 len(train_v["x"]), len(test_v["x"]))

    train_ds = DGTargetDataset(train_v, bc)
    test_ds = DGTargetDataset(test_v, bc)

    model = SupervisedDG(cfg.input_dim, cfg.dg_dim, cfg.k, cfg.t_min, cfg.t_max)
    hist, best = train_supervised(model, train_ds, test_ds, args.epochs,
                                  args.batch_size, args.lr, device,
                                  args.latency_weight,
                                  restore_best=not args.no_restore_best)

    lm, _ = learned_codes(model, test_v["x"], device)
    tgt_for_variants = bc.mask[test_v["base_id"]]
    recall = float((lm * tgt_for_variants).sum() / tgt_for_variants.sum())
    logging.info("\nHeld-out corruption top-k recall vs target: %.3f", recall)

    diag_rows, diag_chance = (None, None)
    if args.diagnostics:
        all_levels = tuple(sorted(set(cfg.corruption_levels)
                                  | set(cfg.heldout_corruption_levels)))
        diag_rows, diag_chance = diagnostics(model, bc, device, all_levels)
        print_table(diag_rows, "DIAGNOSTICS BY CORRUPTION LEVEL")
        print_table([diag_chance], "CHANCE BASELINES")
        print("\n  recall ~ chance at 0% corruption  -> capacity failure "
              "(too many stimuli for this encoder)")
        print("  recall high at 0%, collapsing with corruption -> "
              "noise-robustness failure (no denoising mechanism)")
        print("  identification >> chance while recall is low -> the "
              "neighbourhood survives even when exact membership does not;"
              "\n     this is the metric that matters for downstream routing")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump({"history": hist, "best": best, "heldout_recall": recall,
                   "diagnostics": diag_rows, "chance": diag_chance},
                  f, indent=2)
    logging.info("Results -> %s", args.out_dir)


if __name__ == "__main__":
    main()