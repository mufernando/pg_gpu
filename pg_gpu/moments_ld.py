"""
Integration layer: GPU-accelerated LD statistics for moments inference.

Provides a drop-in replacement for moments.LD.Parsing.compute_ld_statistics()
that uses pg_gpu for the heavy computation, producing output in the exact
format moments expects.

Usage:
    from pg_gpu.moments_ld import compute_ld_statistics

    # Same interface as moments.LD.Parsing.compute_ld_statistics
    ld_stats = compute_ld_statistics(
        vcf_file, rec_map_file=rec_map, pop_file=pop_file,
        pops=["pop0", "pop1"], r_bins=r_bins,
    )

    # Feed directly into moments inference
    mv = moments.LD.Parsing.bootstrap_data({0: ld_stats})
    opt, ll = moments.LD.Inference.optimize_log_lbfgsb(
        p0, [mv["means"], mv["varcovs"]], [model_func], rs=r_bins)
"""

import numpy as np
import cupy as cp
import cupyx
from collections import OrderedDict

from .haplotype_matrix import HaplotypeMatrix
from .haplotype_matrix import (
    _generate_pairs_within_distance,
    _compute_counts_for_pairs,
    _compute_two_pop_statistics_batch,
    _estimate_ld_chunk_size,
)
from . import ld_statistics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ld_statistics(
    vcf_file,
    rec_map_file=None,
    pop_file=None,
    pops=None,
    r_bins=None,
    bp_bins=None,
    use_genotypes=False,
    report=True,
    ac_filter=True,
):
    """GPU-accelerated two-population LD statistics, moments-compatible output.

    Drop-in replacement for moments.LD.Parsing.compute_ld_statistics().

    Parameters
    ----------
    vcf_file : str
        Path to VCF file.
    rec_map_file : str, optional
        Recombination map file (tab-delimited: pos, Map(cM)).
        Required if r_bins is used.
    pop_file : str, optional
        Population assignment file (tab-delimited: sample, pop).
    pops : list of str, optional
        Two population names. Defaults to ['pop0', 'pop1'].
    r_bins : array-like, optional
        Recombination rate bin edges. Pairs are binned by recombination
        distance. Requires rec_map_file.
    bp_bins : array-like, optional
        Base-pair distance bin edges. Alternative to r_bins when no
        recombination map is available.
    use_genotypes : bool
        Ignored (pg_gpu always uses haplotype data). Kept for API compat.
    report : bool
        Print progress messages.
    ac_filter : bool
        Apply biallelic filter before computation.

    Returns
    -------
    dict
        Moments-compatible output with keys:
        - 'bins': list of (bin_lo, bin_hi) tuples
        - 'sums': list of numpy arrays (one per bin + one for het stats)
        - 'stats': (ld_stat_names, het_stat_names)
        - 'pops': population names
    """
    if pops is None:
        pops = ['pop0', 'pop1']
    if len(pops) != 2:
        raise ValueError("Currently only two-population LD is supported")
    if r_bins is None and bp_bins is None:
        raise ValueError("Either r_bins or bp_bins must be provided")

    # Load data
    if report:
        print(f"Loading {vcf_file} ...")
    hm = HaplotypeMatrix.from_vcf(vcf_file)

    # Parse population file and assign sample sets
    if pop_file is not None:
        _assign_populations(hm, vcf_file, pop_file, pops)
    else:
        raise ValueError("pop_file is required for two-population LD")

    # Apply biallelic filter
    if ac_filter:
        hm = hm.apply_biallelic_filter()

    hm.transfer_to_gpu()
    pop1, pop2 = pops

    n_hap = hm.num_haplotypes
    n_var = hm.num_variants
    if report:
        print(f"  {n_hap} haplotypes, {n_var:,} variants")
        print(f"  {pop1}: {len(hm.sample_sets[pop1])} haplotypes, "
              f"{pop2}: {len(hm.sample_sets[pop2])} haplotypes")

    # Determine bin edges and per-variant distances for binning
    if r_bins is not None:
        if rec_map_file is None:
            raise ValueError("rec_map_file required when using r_bins")
        bins = np.asarray(r_bins, dtype=np.float64)
        positions_cpu = hm.positions.get() if hasattr(hm.positions, 'get') else np.asarray(hm.positions)
        gen_dists = _interpolate_genetic_distances(positions_cpu, rec_map_file)
        gen_dists_gpu = cp.asarray(gen_dists)
        # For pair generation, compute conservative max physical distance
        # from the maximum recombination distance and minimum local rate
        max_r = float(bins[-1])
        max_bp_dist = _max_bp_for_r_dist(positions_cpu, gen_dists, max_r)
    else:
        bins = np.asarray(bp_bins, dtype=np.float64)
        gen_dists_gpu = None
        max_bp_dist = float(bins[-1])

    n_bins = len(bins) - 1

    # Compute LD statistics
    if report:
        print(f"  Computing LD statistics ({n_bins} bins) ...")

    ld_sums = _compute_ld_sums(hm, pop1, pop2, bins, gen_dists_gpu,
                                max_bp_dist, ac_filter)

    # Compute heterozygosity statistics
    het_sums = _compute_heterozygosity(hm, pop1, pop2)

    # Assemble moments-compatible output
    ld_names = _ld_stat_names()
    het_names = _het_stat_names()

    bin_tuples = [(float(bins[i]), float(bins[i + 1])) for i in range(n_bins)]
    sums_list = []
    for i in range(n_bins):
        arr = np.array([ld_sums[i, j] for j in range(15)], dtype=np.float64)
        sums_list.append(arr)
    sums_list.append(np.array([
        het_sums['H_0_0'], het_sums['H_0_1'], het_sums['H_1_1']
    ], dtype=np.float64))

    if report:
        print("  Done.")

    return {
        'bins': bin_tuples,
        'sums': sums_list,
        'stats': (ld_names, het_names),
        'pops': pops,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ld_stat_names():
    """15 LD statistic names for two populations (moments convention)."""
    return [
        'DD_0_0', 'DD_0_1', 'DD_1_1',
        'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1',
        'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
        'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1',
        'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1',
    ]


def _het_stat_names():
    """3 heterozygosity statistic names for two populations."""
    return ['H_0_0', 'H_0_1', 'H_1_1']


def _assign_populations(hm, vcf_file, pop_file, pops):
    """Parse population file and set sample_sets on the HaplotypeMatrix."""
    import allel
    vcf = allel.read_vcf(vcf_file, fields=['samples'])
    samples = vcf['samples']
    n_samples = len(samples)

    pop_map = {}
    with open(pop_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] != 'sample':
                pop_map[parts[0]] = parts[1]

    pop_sets = {p: [] for p in pops}
    for i, name in enumerate(samples):
        pop = pop_map.get(name)
        if pop in pop_sets:
            pop_sets[pop].append(i)               # first haplotype
            pop_sets[pop].append(i + n_samples)    # second haplotype

    hm.sample_sets = pop_sets


def _interpolate_genetic_distances(positions, rec_map_file):
    """Load recombination map and interpolate genetic distances at variant positions.

    Returns per-variant cumulative genetic distance in Morgans.
    """
    map_pos = []
    map_vals = []
    with open(rec_map_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    pos = float(parts[0])
                    val = float(parts[1]) / 100.0  # cM to Morgans
                    map_pos.append(pos)
                    map_vals.append(val)
                except ValueError:
                    continue  # skip header

    map_pos = np.array(map_pos)
    map_vals = np.array(map_vals)
    return np.interp(positions, map_pos, map_vals)


def _max_bp_for_r_dist(positions, gen_dists, max_r):
    """Compute conservative max physical distance for a given max recombination distance.

    Uses the minimum local recombination rate to find the largest bp span
    that could correspond to max_r, ensuring no pairs are missed.
    """
    if len(positions) < 2:
        return 1e9
    bp_diffs = np.diff(positions).astype(np.float64)
    r_diffs = np.diff(gen_dists)
    # Avoid division by zero
    valid = bp_diffs > 0
    if not np.any(valid):
        return 1e9
    rates = r_diffs[valid] / bp_diffs[valid]
    min_rate = np.min(rates[rates > 0]) if np.any(rates > 0) else 1e-10
    return max_r / min_rate * 1.1  # 10% safety margin


def _compute_ld_sums(hm, pop1, pop2, bins, gen_dists_gpu, max_bp_dist,
                      ac_filter):
    """Compute LD statistic sums per bin on GPU.

    Uses the same pair-generation and chunked computation pattern as
    HaplotypeMatrix.compute_ld_statistics_gpu_two_pops but bins by
    recombination distance when gen_dists_gpu is provided.
    """
    pos = hm.positions
    if not isinstance(pos, cp.ndarray):
        pos = cp.array(pos)

    n_bins = len(bins) - 1
    bins_gpu = cp.asarray(bins)

    pop1_indices = hm.sample_sets[pop1]
    pop2_indices = hm.sample_sets[pop2]

    chunk_size = _estimate_ld_chunk_size(
        max(len(pop1_indices), len(pop2_indices)))

    # Generate pairs within max physical distance
    idx_i, idx_j = _generate_pairs_within_distance(pos, max_bp_dist)
    total_pairs = len(idx_i)

    if total_pairs == 0:
        return np.zeros((n_bins, 15), dtype=np.float64)

    # Compute distances for binning
    if gen_dists_gpu is not None:
        distances = cp.abs(gen_dists_gpu[idx_j] - gen_dists_gpu[idx_i])
    else:
        distances = pos[idx_j] - pos[idx_i]

    bin_inds = cp.digitize(distances, bins_gpu) - 1

    bin_sums = cp.zeros((n_bins, 15), dtype=cp.float64)

    for chunk_start in range(0, total_pairs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pairs)

        chunk_idx_i = idx_i[chunk_start:chunk_end]
        chunk_idx_j = idx_j[chunk_start:chunk_end]
        chunk_bin_inds = bin_inds[chunk_start:chunk_end]

        counts_pop1, n_valid1 = _compute_counts_for_pairs(
            hm.haplotypes, chunk_idx_i, chunk_idx_j, pop1_indices)
        counts_pop2, n_valid2 = _compute_counts_for_pairs(
            hm.haplotypes, chunk_idx_i, chunk_idx_j, pop2_indices)

        chunk_stats = _compute_two_pop_statistics_batch(
            counts_pop1, counts_pop2, n_valid1, n_valid2, ld_statistics)

        valid_mask = (chunk_bin_inds >= 0) & (chunk_bin_inds < n_bins)
        valid_bin_inds = chunk_bin_inds[valid_mask]
        valid_stats = chunk_stats[valid_mask]

        for stat_idx in range(15):
            cupyx.scatter_add(bin_sums[:, stat_idx], valid_bin_inds,
                              valid_stats[:, stat_idx])

        del counts_pop1, counts_pop2, n_valid1, n_valid2, chunk_stats

    return bin_sums.get()


def _compute_heterozygosity(hm, pop1, pop2):
    """Compute H_0_0, H_0_1, H_1_1 on GPU.

    Matches moments' heterozygosity computation (Parsing.py lines 1033-1058).
    """
    pop1_idx = hm.sample_sets[pop1]
    pop2_idx = hm.sample_sets[pop2]
    hap = hm.haplotypes  # (n_hap, n_var) on GPU

    # Allele counts per population per site
    pop1_haps = hap[pop1_idx, :]
    pop2_haps = hap[pop2_idx, :]

    pop1_alt = cp.sum(cp.maximum(pop1_haps, 0).astype(cp.int32), axis=0).astype(cp.float64)
    pop1_n = cp.float64(len(pop1_idx))
    pop1_ref = pop1_n - pop1_alt

    pop2_alt = cp.sum(cp.maximum(pop2_haps, 0).astype(cp.int32), axis=0).astype(cp.float64)
    pop2_n = cp.float64(len(pop2_idx))
    pop2_ref = pop2_n - pop2_alt

    # Within-pop heterozygosity: H = sum(2 * ref * alt / (n * (n-1)))
    H_0_0 = float(cp.sum(2.0 * pop1_ref * pop1_alt / (pop1_n * (pop1_n - 1))).get())
    H_1_1 = float(cp.sum(2.0 * pop2_ref * pop2_alt / (pop2_n * (pop2_n - 1))).get())

    # Cross-pop: H = sum((ref1*alt2 + alt1*ref2) / (n1 * n2))
    H_0_1 = float(cp.sum(
        (pop1_ref * pop2_alt + pop1_alt * pop2_ref) / (pop1_n * pop2_n)
    ).get())

    return {'H_0_0': H_0_0, 'H_0_1': H_0_1, 'H_1_1': H_1_1}
