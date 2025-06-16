#!/usr/bin/env python
"""Compare polymorphism statistics between pg_gpu and scikit-allel with proper normalization."""

import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity

def compare_statistics_normalized():
    """Compare diversity statistics with normalization."""
    
    print("=" * 80)
    print("Comparing pg_gpu vs scikit-allel (with normalization)")
    print("=" * 80)
    
    # Test case with specific positions to control span
    haplotypes = np.array([
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.int8)
    
    # Use specific positions with known span
    positions = np.array([100, 200, 300, 400, 500])
    start_pos = positions[0]
    end_pos = positions[-1]
    span = end_pos - start_pos + 1  # 500 - 100 + 1 = 401 bases
    
    print(f"\nTest dataset:")
    print(f"  Shape: {haplotypes.shape[0]} haplotypes, {haplotypes.shape[1]} variants")
    print(f"  Positions: {positions}")
    print(f"  Genomic span: {span} bases (from {start_pos} to {end_pos})")
    
    # Create pg_gpu HaplotypeMatrix
    matrix = HaplotypeMatrix(haplotypes, positions, start_pos, end_pos)
    
    # Create scikit-allel HaplotypeArray
    h = allel.HaplotypeArray(haplotypes.T)
    ac = h.count_alleles()
    
    print("\n" + "-" * 60)
    print("1. Pi (Nucleotide Diversity):")
    print("-" * 60)
    
    # pg_gpu - raw value
    pi_pg_raw = diversity.pi(matrix, span_normalize=False)
    print(f"  pg_gpu (raw):              {pi_pg_raw:.6f}")
    
    # pg_gpu - normalized by span manually
    pi_pg_normalized = pi_pg_raw / span
    print(f"  pg_gpu (normalized):       {pi_pg_normalized:.6f}")
    
    # pg_gpu - using span_normalize=True
    pi_pg_auto = diversity.pi(matrix, span_normalize=True)
    print(f"  pg_gpu (span_normalize):   {pi_pg_auto:.6f}")
    
    # scikit-allel
    pi_allel = allel.sequence_diversity(positions, ac, start=start_pos, stop=end_pos)
    print(f"  scikit-allel:              {pi_allel:.6f}")
    
    print(f"\n  Difference (normalized):   {abs(pi_pg_normalized - pi_allel):.8f}")
    print(f"  Match: {np.isclose(pi_pg_normalized, pi_allel, rtol=1e-6)}")
    
    print("\n" + "-" * 60)
    print("2. Watterson's Theta:")
    print("-" * 60)
    
    # pg_gpu - raw value
    theta_pg_raw = diversity.theta_w(matrix, span_normalize=False)
    print(f"  pg_gpu (raw):              {theta_pg_raw:.6f}")
    
    # pg_gpu - normalized
    theta_pg_normalized = theta_pg_raw / span
    print(f"  pg_gpu (normalized):       {theta_pg_normalized:.6f}")
    
    # pg_gpu - using span_normalize=True
    theta_pg_auto = diversity.theta_w(matrix, span_normalize=True)
    print(f"  pg_gpu (span_normalize):   {theta_pg_auto:.6f}")
    
    # scikit-allel
    theta_allel = allel.watterson_theta(positions, ac, start=start_pos, stop=end_pos)
    print(f"  scikit-allel:              {theta_allel:.6f}")
    
    print(f"\n  Difference (normalized):   {abs(theta_pg_normalized - theta_allel):.8f}")
    print(f"  Match: {np.isclose(theta_pg_normalized, theta_allel, rtol=1e-6)}")
    
    print("\n" + "-" * 60)
    print("3. Tajima's D (no normalization needed):")
    print("-" * 60)
    
    # pg_gpu
    tajd_pg = diversity.tajimas_d(matrix)
    print(f"  pg_gpu:       {tajd_pg:.6f}")
    
    # scikit-allel
    tajd_allel = allel.tajima_d(ac, pos=positions)
    print(f"  scikit-allel: {tajd_allel:.6f}")
    
    print(f"\n  Difference:   {abs(tajd_pg - tajd_allel):.8f}")
    print(f"  Match: {np.isclose(tajd_pg, tajd_allel, rtol=1e-6)}")
    
    # Test with a larger dataset to verify
    print("\n" + "=" * 80)
    print("Testing with larger random dataset:")
    print("=" * 80)
    
    n_haplotypes = 50
    n_variants = 100
    
    # Generate random haplotypes
    np.random.seed(42)  # For reproducibility
    haplotypes = np.random.randint(0, 2, size=(n_haplotypes, n_variants), dtype=np.int8)
    
    # Create positions with 1kb spacing
    positions = np.arange(1, n_variants + 1) * 1000
    start_pos = positions[0]
    end_pos = positions[-1]
    span = end_pos - start_pos + 1
    
    print(f"\nDataset: {n_haplotypes} haplotypes, {n_variants} variants")
    print(f"Span: {span} bases")
    
    # pg_gpu
    matrix = HaplotypeMatrix(haplotypes, positions, start_pos, end_pos)
    
    # scikit-allel
    h = allel.HaplotypeArray(haplotypes.T)
    ac = h.count_alleles()
    
    # Compare all statistics
    stats = {
        'pi': {
            'pg_gpu': diversity.pi(matrix, span_normalize=True),
            'allel': allel.sequence_diversity(positions, ac, start=start_pos, stop=end_pos)
        },
        'theta_w': {
            'pg_gpu': diversity.theta_w(matrix, span_normalize=True),
            'allel': allel.watterson_theta(positions, ac, start=start_pos, stop=end_pos)
        },
        'tajimas_d': {
            'pg_gpu': diversity.tajimas_d(matrix),
            'allel': allel.tajima_d(ac, pos=positions)
        }
    }
    
    print("\nResults:")
    for stat_name, values in stats.items():
        pg_val = values['pg_gpu']
        allel_val = values['allel']
        diff = abs(pg_val - allel_val)
        match = np.isclose(pg_val, allel_val, rtol=1e-6) if not np.isnan(allel_val) else False
        
        print(f"\n{stat_name}:")
        print(f"  pg_gpu:       {pg_val:.8f}")
        print(f"  scikit-allel: {allel_val:.8f}")
        print(f"  Difference:   {diff:.8f}")
        print(f"  Match:        {match}")


if __name__ == "__main__":
    compare_statistics_normalized()