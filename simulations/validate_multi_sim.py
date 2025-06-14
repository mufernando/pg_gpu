#!/usr/bin/env python3
"""
Validate two-population LD statistics between pg_gpu and moments packages
using multiple simulations with different random seeds.

This script:
1. Runs 3 simulations in parallel with different seeds
2. Computes LD statistics using both moments and pg_gpu with detailed timing
3. Creates validation plots comparing statistical results
4. Creates comprehensive timing analysis plots showing performance differences
5. Saves timing data to CSV for further analysis

Features:
- Parallel simulation execution for efficiency
- Detailed timing breakdown (setup vs computation for GPU)
- Performance scaling analysis
- Speedup calculations and visualization
- Caching support to avoid recomputation

Output files:
- simulations/multi_sim_validation.png: Statistical validation plots
- simulations/timing_comparison.png: Performance comparison plots
- simulations/timing_results.csv: Raw timing data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moments'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil
import time
import pandas as pd
import argparse

# Import the required modules
import moments.LD.Parsing as mParsing
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import allel
import msprime
import demes


def simulate_im_model(seed, output_dir):
    """
    Simulate IM model with given seed.
    Returns paths to VCF and population files.
    """
    print(f"Simulating with seed {seed}...")
    
    # Simulation parameters (matching simulate_im_vcf.py)
    L = 1e6
    u = r = 1.5e-8
    n = 10
    
    # Load demographic model
    g = demes.load("data/demes_mod.yaml")
    demog = msprime.Demography.from_demes(g)
    
    # Run simulation
    trees = msprime.sim_ancestry(
        {"deme0": n, "deme1": n},
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
        random_seed=seed,
    )
    
    trees = msprime.sim_mutations(trees, rate=u, random_seed=seed + 1000)
    
    # Write VCF
    vcf_path = os.path.join(output_dir, f"sim_seed{seed}.vcf")
    with open(vcf_path, "w") as fout:
        trees.write_vcf(fout)
    
    # Create population file
    pop_file = os.path.join(output_dir, f"sim_seed{seed}_pops.txt")
    with open(pop_file, "w") as f:
        f.write("sample\tpop\n")
        for i in range(n):
            f.write(f"tsk_{i}\tdeme0\n")
        for i in range(n, 2*n):
            f.write(f"tsk_{i}\tdeme1\n")
    
    print(f"Simulation with seed {seed} complete")
    return vcf_path, pop_file, trees


def run_validation_for_seed(seed, output_dir, bp_bins, use_cache=True):
    """
    Run validation for a single simulation.
    Returns moments and GPU statistics along with timing information.
    """
    # Run simulation
    sim_start = time.time()
    vcf_path, pop_file, trees = simulate_im_model(seed, output_dir)
    sim_time = time.time() - sim_start
    
    # Get dataset size info
    vcf = allel.read_vcf(vcf_path)
    n_variants = len(vcf['variants/POS'])
    n_samples = vcf['samples'].shape[0]
    n_pairs = n_variants * (n_variants - 1) // 2
    
    print(f"Seed {seed}: {n_variants} variants, {n_samples} samples, {n_pairs:,} variant pairs")
    
    # Define populations
    pops = ["deme0", "deme1"]
    
    # Cache setup
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = f"seed{seed}_{len(bp_bins)}bins"
    cache_file = cache_dir / f"moments_ld_{cache_key}.pkl"
    
    # Run moments LD calculation with timing
    moments_time = None
    if use_cache and cache_file.exists():
        print(f"Loading cached moments results for seed {seed}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                moments_stats, moments_time = cached_data
            else:
                moments_stats = cached_data
                moments_time = None
    else:
        print(f"Computing moments LD for seed {seed}...")
        moments_start = time.time()
        moments_stats = mParsing.compute_ld_statistics(
            vcf_path,
            pop_file=pop_file,
            pops=pops,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        moments_time = time.time() - moments_start
        print(f"Moments computation took {moments_time:.2f} seconds")
        
        # Cache results with timing
        with open(cache_file, 'wb') as f:
            pickle.dump((moments_stats, moments_time), f)
    
    # Run GPU LD calculation with timing
    print(f"Computing GPU LD for seed {seed}...")
    
    # Time VCF loading and setup
    setup_start = time.time()
    h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
    
    # Set up population assignments
    pop_assignments = {}
    with open(pop_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            sample, pop = line.strip().split()
            pop_assignments[sample] = pop
    
    # Create sample sets
    pop_sets = {"deme0": [], "deme1": []}
    for i, sample_name in enumerate(vcf['samples']):
        pop = pop_assignments.get(sample_name, None)
        if pop in pop_sets:
            pop_sets[pop].append(i)
            pop_sets[pop].append(i + n_samples)
    
    h_gpu.sample_sets = pop_sets
    setup_time = time.time() - setup_start
    
    # Time the actual computation
    gpu_start = time.time()
    gpu_stats = h_gpu.compute_ld_statistics_gpu_two_pops(
        bp_bins=bp_bins,
        pop1="deme0",
        pop2="deme1",
        missing=False,
        raw=True
    )
    gpu_comp_time = time.time() - gpu_start
    gpu_total_time = setup_time + gpu_comp_time
    
    print(f"GPU setup took {setup_time:.2f} seconds")
    print(f"GPU computation took {gpu_comp_time:.2f} seconds")
    print(f"GPU total time: {gpu_total_time:.2f} seconds")
    
    # Create timing dictionary
    timing_info = {
        'seed': seed,
        'n_variants': n_variants,
        'n_samples': n_samples,
        'n_pairs': n_pairs,
        'simulation_time': sim_time,
        'moments_time': moments_time,
        'gpu_setup_time': setup_time,
        'gpu_computation_time': gpu_comp_time,
        'gpu_total_time': gpu_total_time
    }
    
    return seed, moments_stats, gpu_stats, timing_info


def extract_statistics_for_plotting(moments_stats, gpu_stats):
    """
    Extract statistics from moments and GPU results for plotting.
    Returns two arrays of corresponding values.
    """
    stat_names = moments_stats['stats'][0]
    
    moments_values = []
    gpu_values = []
    
    for bin_range, moments_sums in zip(moments_stats['bins'], moments_stats['sums']):
        gpu_bin = gpu_stats[bin_range]
        
        for stat_name, mom_val in zip(stat_names, moments_sums):
            gpu_val = gpu_bin[stat_name]
            moments_values.append(mom_val)
            gpu_values.append(gpu_val)
    
    return np.array(moments_values), np.array(gpu_values)


def create_timing_comparison_figure(timing_data, output_file):
    """
    Create figure comparing timing performance between moments and GPU.
    """
    df = pd.DataFrame(timing_data)
    
    # Filter out cases where moments_time is None (cached results)
    df_with_timing = df[df['moments_time'].notna()].copy()
    
    if len(df_with_timing) == 0:
        print("No timing data available (all results were cached)")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Absolute timing comparison
    seeds = df_with_timing['seed']
    x_pos = np.arange(len(seeds))
    width = 0.35
    
    moments_times = df_with_timing['moments_time']
    gpu_times = df_with_timing['gpu_total_time']
    
    bars1 = ax1.bar(x_pos - width/2, moments_times, width, label='Moments', color='orange', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, gpu_times, width, label='GPU Total', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Simulation Seed')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Absolute Computing Time Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(seeds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (m_time, g_time) in enumerate(zip(moments_times, gpu_times)):
        ax1.text(i - width/2, m_time + max(moments_times) * 0.01, f'{m_time:.1f}s', 
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, g_time + max(gpu_times) * 0.01, f'{g_time:.1f}s', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Speedup ratio
    speedup = moments_times / gpu_times
    bars = ax2.bar(x_pos, speedup, color='green', alpha=0.7)
    ax2.set_xlabel('Simulation Seed')
    ax2.set_ylabel('Speedup Factor (Moments time / GPU time)')
    ax2.set_title('GPU Speedup vs Moments')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seeds)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.legend()
    
    # Add value labels
    for i, speed in enumerate(speedup):
        ax2.text(i, speed + max(speedup) * 0.01, f'{speed:.1f}x', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: GPU timing breakdown
    gpu_setup_times = df_with_timing['gpu_setup_time']
    gpu_comp_times = df_with_timing['gpu_computation_time']
    
    bars1 = ax3.bar(x_pos, gpu_setup_times, width, label='Setup', color='lightblue', alpha=0.7)
    bars2 = ax3.bar(x_pos, gpu_comp_times, width, bottom=gpu_setup_times, 
                   label='Computation', color='darkblue', alpha=0.7)
    
    ax3.set_xlabel('Simulation Seed')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('GPU Time Breakdown')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(seeds)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance vs dataset size
    n_pairs = df_with_timing['n_pairs']
    
    # Plot both on same axis with different scales
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(n_pairs, moments_times, 'o-', color='orange', linewidth=2, 
                    markersize=8, label='Moments')
    line2 = ax4_twin.plot(n_pairs, gpu_times, 's-', color='blue', linewidth=2, 
                         markersize=8, label='GPU')
    
    ax4.set_xlabel('Number of Variant Pairs')
    ax4.set_ylabel('Moments Time (seconds)', color='orange')
    ax4_twin.set_ylabel('GPU Time (seconds)', color='blue')
    ax4.set_title('Performance vs Dataset Size')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis to show large numbers nicely
    ax4.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.suptitle('Performance Comparison: Moments vs GPU Implementation', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Timing comparison figure saved to: {output_file}")
    plt.close()


def create_comparison_figure(results, output_file):
    """
    Create figure with 3 scatter plots comparing moments vs GPU statistics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Color map for different statistics
    color_map = {
        'DD': 'blue',
        'Dz': 'green',
        'pi2': 'red'
    }
    
    for idx, (seed, moments_stats, gpu_stats, timing_info) in enumerate(results):
        ax = axes[idx]
        
        # Extract values
        moments_vals, gpu_vals = extract_statistics_for_plotting(moments_stats, gpu_stats)
        
        # Get statistic names for coloring
        stat_names = moments_stats['stats'][0]
        n_stats = len(stat_names)
        n_bins = len(moments_stats['bins'])
        
        # Create colors array
        colors = []
        for bin_idx in range(n_bins):
            for stat_name in stat_names:
                if stat_name.startswith('DD'):
                    colors.append(color_map['DD'])
                elif stat_name.startswith('Dz'):
                    colors.append(color_map['Dz'])
                else:  # pi2
                    colors.append(color_map['pi2'])
        
        # Create scatter plot
        ax.scatter(moments_vals, gpu_vals, c=colors, alpha=0.6, s=30)
        
        # Add diagonal line
        min_val = min(moments_vals.min(), gpu_vals.min())
        max_val = max(moments_vals.max(), gpu_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Set scales and labels
        ax.set_xscale('symlog')
        ax.set_yscale('symlog')
        ax.set_xlabel('Moments Statistics')
        if idx == 0:
            ax.set_ylabel('GPU Statistics')
        ax.set_title(f'Seed {seed}')
        ax.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        correlation = np.corrcoef(moments_vals, gpu_vals)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend to the last subplot
    dd_patch = mpatches.Patch(color='blue', label='DD statistics')
    dz_patch = mpatches.Patch(color='green', label='Dz statistics')
    pi2_patch = mpatches.Patch(color='red', label='π₂ statistics')
    axes[-1].legend(handles=[dd_patch, dz_patch, pi2_patch], 
                    loc='lower right', framealpha=0.9)
    
    plt.suptitle('Correspondence between Moments and GPU LD Statistics\nIM Model with Different Random Seeds', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Validate and time LD statistics between moments and GPU implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_multi_sim.py                    # Run with default settings
  python validate_multi_sim.py --seeds 1 2 3     # Use custom seeds
  python validate_multi_sim.py --no-cache        # Force recomputation
  python validate_multi_sim.py --workers 2       # Use 2 parallel workers
        """
    )
    
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[12345, 67890, 11111],
                       help='Random seeds for simulations (default: 12345 67890 11111)')
    parser.add_argument('--bins', type=int, default=6,
                       help='Number of distance bins (default: 6)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3)')
    parser.add_argument('--output-dir', default="simulations/output",
                       help='Output directory (default: simulations/output)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching, force recomputation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Define parameters
    seeds = args.seeds
    bp_bins = np.logspace(2, 6, args.bins)
    use_cache = not args.no_cache
    
    print("=" * 60)
    print("MULTI-SIMULATION VALIDATION WITH TIMING ANALYSIS")
    print("=" * 60)
    print(f"Running {len(seeds)} simulations with different random seeds")
    print(f"Seeds: {seeds}")
    print(f"Distance bins: {bp_bins}")
    
    # Run validations in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_seed = {
            executor.submit(run_validation_for_seed, seed, output_dir, bp_bins, use_cache): seed 
            for seed in seeds
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed validation for seed {seed}")
            except Exception as e:
                print(f"Validation for seed {seed} failed with error: {e}")
    
    # Sort results by seed to maintain consistent ordering
    results.sort(key=lambda x: x[0])
    
    # Extract timing data
    timing_data = [result[3] for result in results]  # timing_info is 4th element
    
    # Create comparison figure
    output_file = "simulations/multi_sim_validation.png"
    create_comparison_figure(results, output_file)
    
    # Create timing comparison figure
    timing_output_file = "simulations/timing_comparison.png"
    create_timing_comparison_figure(timing_data, timing_output_file)
    
    # Save timing data to CSV for further analysis
    timing_df = pd.DataFrame(timing_data)
    timing_csv_file = "simulations/timing_results.csv"
    timing_df.to_csv(timing_csv_file, index=False)
    print(f"Timing data saved to: {timing_csv_file}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for seed, moments_stats, gpu_stats, timing_info in results:
        moments_vals, gpu_vals = extract_statistics_for_plotting(moments_stats, gpu_stats)
        
        # Calculate errors
        relative_errors = np.abs(gpu_vals - moments_vals) / (np.abs(moments_vals) + 1e-10)
        
        print(f"\nSeed {seed}:")
        print(f"  Dataset: {timing_info['n_variants']} variants, {timing_info['n_pairs']:,} pairs")
        print(f"  Overall correlation: {np.corrcoef(moments_vals, gpu_vals)[0, 1]:.6f}")
        print(f"  Mean relative error: {relative_errors.mean():.6f}")
        print(f"  Max relative error: {relative_errors.max():.6f}")
        print(f"  Median relative error: {np.median(relative_errors):.6f}")
    
    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Filter for non-cached results
    timing_df_filtered = timing_df[timing_df['moments_time'].notna()]
    
    if len(timing_df_filtered) > 0:
        print("\nPerformance Summary (non-cached results only):")
        print("-" * 50)
        
        # Calculate speedups
        speedups = timing_df_filtered['moments_time'] / timing_df_filtered['gpu_total_time']
        
        # Print individual results
        for _, row in timing_df_filtered.iterrows():
            speedup = row['moments_time'] / row['gpu_total_time']
            setup_pct = (row['gpu_setup_time'] / row['gpu_total_time']) * 100
            comp_pct = (row['gpu_computation_time'] / row['gpu_total_time']) * 100
            
            print(f"\nSeed {int(row['seed'])}:")
            print(f"  Moments time:     {row['moments_time']:.2f} seconds")
            print(f"  GPU total time:   {row['gpu_total_time']:.2f} seconds")
            print(f"    - Setup:        {row['gpu_setup_time']:.2f}s ({setup_pct:.1f}%)")
            print(f"    - Computation:  {row['gpu_computation_time']:.2f}s ({comp_pct:.1f}%)")
            print(f"  Speedup:          {speedup:.1f}x")
        
        # Overall statistics
        print(f"\nOverall Performance:")
        print(f"  Average speedup:  {speedups.mean():.1f}x")
        print(f"  Min speedup:      {speedups.min():.1f}x")
        print(f"  Max speedup:      {speedups.max():.1f}x")
        
        # GPU breakdown analysis
        avg_setup_pct = (timing_df_filtered['gpu_setup_time'] / timing_df_filtered['gpu_total_time']).mean() * 100
        avg_comp_pct = (timing_df_filtered['gpu_computation_time'] / timing_df_filtered['gpu_total_time']).mean() * 100
        
        print(f"\nGPU Time Breakdown (average):")
        print(f"  Setup:        {avg_setup_pct:.1f}%")
        print(f"  Computation:  {avg_comp_pct:.1f}%")
        
        # Performance scaling analysis
        if len(timing_df_filtered) > 1:
            # Simple correlation analysis
            pairs_to_moments_corr = timing_df_filtered[['n_pairs', 'moments_time']].corr().iloc[0, 1]
            pairs_to_gpu_corr = timing_df_filtered[['n_pairs', 'gpu_total_time']].corr().iloc[0, 1]
            
            print(f"\nScaling Analysis:")
            print(f"  Pairs vs Moments time correlation: {pairs_to_moments_corr:.3f}")
            print(f"  Pairs vs GPU time correlation:     {pairs_to_gpu_corr:.3f}")
    else:
        print("\nNo timing data available (all results were cached)")
        print("To get timing data, delete the cache directory and re-run:")
        print(f"  rm -rf {output_dir}/cache")
    
    print(f"\nValidation complete!")
    print(f"Results saved to:")
    print(f"  Validation plots: {output_file}")
    if len(timing_df_filtered) > 0:
        print(f"  Timing plots:     {timing_output_file}")
    print(f"  Timing data:      {timing_csv_file}")


if __name__ == "__main__":
    main()