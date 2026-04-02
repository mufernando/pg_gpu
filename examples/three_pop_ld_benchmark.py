#!/usr/bin/env python
"""
Benchmark: pg_gpu three-population LD statistics vs moments.

Validates that pg_gpu produces the same LD statistics (DD, Dz, pi2)
as moments for a three-population model, and compares wall-clock time.

Usage:
    pixi run -e moments python examples/three_pop_ld_benchmark.py
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import msprime
import moments.LD.Parsing as mParsing
from pg_gpu.moments_ld import compute_ld_statistics

CACHE_DIR = Path("cache")
POPS = ["pop0", "pop1", "pop2"]
BP_BINS = np.logspace(2, 6, 6)

# Simulation parameters
N_SAMPLES = 10
SEQ_LEN = 500_000
REC_RATE = 1e-8
MUT_RATE = 1e-7
SEED = 42


def simulate_data():
    """Simulate a 3-population tree and write VCF + pop file."""
    CACHE_DIR.mkdir(exist_ok=True)
    vcf_path = CACHE_DIR / "three_pop_benchmark.vcf"
    pop_path = CACHE_DIR / "three_pop_benchmark_pops.txt"

    if vcf_path.exists() and pop_path.exists():
        print("  Simulation: loaded from cache")
        return str(vcf_path), str(pop_path)

    demography = msprime.Demography()
    demography.add_population(name="pop0", initial_size=1000)
    demography.add_population(name="pop1", initial_size=1000)
    demography.add_population(name="pop2", initial_size=1000)
    demography.add_population(name="anc01", initial_size=2000)
    demography.add_population(name="anc012", initial_size=2000)
    demography.add_population_split(
        time=500, derived=["pop0", "pop1"], ancestral="anc01")
    demography.add_population_split(
        time=1000, derived=["anc01", "pop2"], ancestral="anc012")

    samples = {p: N_SAMPLES for p in POPS}
    ts = msprime.sim_ancestry(
        samples=samples, demography=demography,
        sequence_length=SEQ_LEN, recombination_rate=REC_RATE,
        random_seed=SEED)
    ts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=SEED)

    with open(vcf_path, 'w') as f:
        ts.write_vcf(f, allow_position_zero=True)

    with open(pop_path, 'w') as f:
        f.write("sample\tpop\n")
        for ind in ts.individuals():
            pop_name = ts.population(ind.population).metadata.get(
                'name', f"pop{ind.population}")
            f.write(f"tsk_{ind.id}\t{pop_name}\n")

    n_vars = ts.num_sites
    print(f"  Simulated {n_vars:,} variants, {N_SAMPLES} samples x {len(POPS)} pops")
    return str(vcf_path), str(pop_path)


def run_moments(vcf_path, pop_path, use_cache=True):
    """Run moments LD computation (with disk cache)."""
    cache_file = CACHE_DIR / "moments_ld_3pop.pkl"

    if use_cache and cache_file.exists():
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        print("  moments: loaded from cache")
        return result['stats'], result['time']

    t0 = time.time()
    ld_stats = mParsing.compute_ld_statistics(
        vcf_path, pop_file=pop_path, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=False, report=False)
    elapsed = time.time() - t0

    with open(cache_file, 'wb') as f:
        pickle.dump({'stats': ld_stats, 'time': elapsed}, f)

    print(f"  moments: computed in {elapsed:.1f}s (cached for next run)")
    return ld_stats, elapsed


def run_pg_gpu(vcf_path, pop_path):
    """Run pg_gpu LD computation (VCF pre-loaded, only GPU time measured)."""
    from pg_gpu.haplotype_matrix import HaplotypeMatrix

    # Pre-load data (not timed, matching 2-pop benchmark convention)
    hm = HaplotypeMatrix.from_vcf(vcf_path)
    hm.load_pop_file(pop_path, pops=POPS)
    hm = hm.apply_biallelic_filter()
    hm.transfer_to_gpu()

    # Warmup
    _ = compute_ld_statistics(
        pops=POPS, bp_bins=BP_BINS, report=False,
        haplotype_matrix=hm, ac_filter=False)

    t0 = time.time()
    gpu_stats = compute_ld_statistics(
        pops=POPS, bp_bins=BP_BINS, report=False,
        haplotype_matrix=hm, ac_filter=False)
    elapsed = time.time() - t0

    print(f"  pg_gpu: computed in {elapsed:.3f}s")
    return gpu_stats, elapsed


def compare(moments_stats, gpu_stats):
    """Compare moments and pg_gpu results."""
    stat_names = moments_stats['stats'][0]

    all_moments = []
    all_gpu = []
    all_labels = []

    for bin_idx in range(len(moments_stats['bins'])):
        mom_sums = moments_stats['sums'][bin_idx]
        gpu_sums = gpu_stats['sums'][bin_idx]

        for s, mv, gv in zip(stat_names, mom_sums, gpu_sums):
            all_moments.append(mv)
            all_gpu.append(gv)
            all_labels.append(s)

    mom = np.array(all_moments)
    gpu = np.array(all_gpu)
    rel_err = np.abs(gpu - mom) / (np.abs(mom) + 1e-10)

    print(f"\n  {len(mom)} statistic-bin pairs compared")
    print(f"  Correlation: {np.corrcoef(mom, gpu)[0,1]:.8f}")
    print(f"  Mean relative error: {rel_err.mean():.2e}")
    print(f"  Max relative error:  {rel_err.max():.2e}")
    print(f"  Pairs with <1% error: {(rel_err < 0.01).sum()}/{len(rel_err)}")

    return mom, gpu, all_labels, stat_names


def plot_results(mom, gpu, labels, stat_names, t_moments, t_pg):
    """Create a two-panel figure: correspondence plot + timing comparison."""
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              gridspec_kw={'width_ratios': [3, 2]})

    # Panel A: correspondence
    ax = axes[0]
    colors = []
    for lab in labels:
        if lab.startswith('DD'):
            colors.append('#3498db')
        elif lab.startswith('Dz'):
            colors.append('#2ecc71')
        else:
            colors.append('#e74c3c')

    ax.scatter(mom, gpu, c=colors, alpha=0.7, s=40, edgecolors='0.3', linewidths=0.3)
    lo = min(mom.min(), gpu.min())
    hi = max(mom.max(), gpu.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'k--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel("moments")
    ax.set_ylabel("pg_gpu")
    ax.set_title("Three-population LD statistics")
    ax.set_xscale('symlog', linthresh=1e-6)
    ax.set_yscale('symlog', linthresh=1e-6)
    ticks = [-1e-2, -1e-4, 0, 1e-4, 1e-2, 1, 1e2]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#3498db', label='DD'),
                       Patch(color='#2ecc71', label='Dz'),
                       Patch(color='#e74c3c', label='pi2')],
              loc='upper left', fontsize=10)

    # Panel B: timing
    ax2 = axes[1]
    tools = ['moments', 'pg_gpu']
    times = [t_moments, t_pg]
    bars = ax2.bar(tools, times, color=['#e74c3c', '#2ecc71'],
                   edgecolor='0.3', linewidth=0.5)
    ax2.set_ylabel("Wall time (s)")
    ax2.set_title(f"Speedup: {t_moments/t_pg:.0f}x")
    ax2.set_yscale('log')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, t * 1.2,
                 f"{t:.2f}s", ha='center', va='bottom', fontsize=11)

    n_bins = len(mom) // len(stat_names)
    fig.suptitle("pg_gpu vs moments: three-population LD (DD, Dz, pi2)\n"
                 f"3-pop split model, {len(stat_names)} statistics x {n_bins} distance bins",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig("examples/three_pop_ld_benchmark.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to examples/three_pop_ld_benchmark.png")


def main():
    print("Three-population LD benchmark: pg_gpu vs moments")
    print("=" * 55)

    print("\nSimulating data ...")
    vcf_path, pop_path = simulate_data()

    print("\nComputing LD statistics ...")
    moments_stats, t_moments = run_moments(vcf_path, pop_path)
    gpu_stats, t_pg = run_pg_gpu(vcf_path, pop_path)

    print("\nValidation:")
    mom, gpu, labels, stat_names = compare(moments_stats, gpu_stats)

    print("\nTiming:")
    print(f"  moments: {t_moments:.2f}s")
    print(f"  pg_gpu:  {t_pg:.3f}s")
    print(f"  Speedup: {t_moments/t_pg:.0f}x")

    plot_results(mom, gpu, labels, stat_names, t_moments, t_pg)


if __name__ == "__main__":
    main()
