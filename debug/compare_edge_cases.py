#!/usr/bin/env python
"""Compare edge cases between pg_gpu and scikit-allel."""

import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity

def test_edge_cases():
    """Test edge cases and special scenarios."""
    
    print("=" * 80)
    print("Testing Edge Cases")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "No variation (all sites fixed)",
            "haplotypes": np.zeros((10, 5), dtype=np.int8),
            "positions": np.array([100, 200, 300, 400, 500])
        },
        {
            "name": "All singletons",
            "haplotypes": np.eye(10, 10, dtype=np.int8),  # Identity matrix
            "positions": np.arange(10) * 100 + 100
        },
        {
            "name": "Single segregating site",
            "haplotypes": np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ], dtype=np.int8),
            "positions": np.array([100, 200, 300])
        },
        {
            "name": "Two populations - one fixed difference",
            "haplotypes": np.array([
                [0, 0, 1],  # Pop1
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 0],  # Pop2
                [1, 0, 0],
                [1, 0, 0]
            ], dtype=np.int8),
            "positions": np.array([100, 200, 300])
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Test: {test_case['name']}")
        print('=' * 60)
        
        haplotypes = test_case['haplotypes']
        positions = test_case['positions']
        n_haplotypes, n_variants = haplotypes.shape
        
        print(f"Shape: {n_haplotypes} haplotypes, {n_variants} variants")
        print(f"Haplotype matrix:\n{haplotypes}")
        
        # Create data structures
        start_pos = positions[0]
        end_pos = positions[-1]
        span = end_pos - start_pos + 1
        
        matrix = HaplotypeMatrix(haplotypes, positions, start_pos, end_pos)
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        
        print(f"\nAllele counts:")
        for i, pos in enumerate(positions):
            print(f"  Position {pos}: {ac[i]}")
        
        # Compare statistics
        print("\nStatistics comparison:")
        
        # Segregating sites
        seg_pg = diversity.segregating_sites(matrix)
        seg_allel = np.sum(ac.is_segregating())
        print(f"\nSegregating sites:")
        print(f"  pg_gpu:       {seg_pg}")
        print(f"  scikit-allel: {seg_allel}")
        print(f"  Match:        {seg_pg == seg_allel}")
        
        # Pi
        pi_pg = diversity.pi(matrix, span_normalize=True)
        pi_allel = allel.sequence_diversity(positions, ac, start=start_pos, stop=end_pos)
        print(f"\nPi (normalized):")
        print(f"  pg_gpu:       {pi_pg:.8f}")
        print(f"  scikit-allel: {pi_allel:.8f}")
        print(f"  Match:        {np.isclose(pi_pg, pi_allel, rtol=1e-6)}")
        
        # Theta
        theta_pg = diversity.theta_w(matrix, span_normalize=True)
        theta_allel = allel.watterson_theta(positions, ac, start=start_pos, stop=end_pos)
        print(f"\nTheta_w (normalized):")
        print(f"  pg_gpu:       {theta_pg:.8f}")
        print(f"  scikit-allel: {theta_allel:.8f}")
        print(f"  Match:        {np.isclose(theta_pg, theta_allel, rtol=1e-6)}")
        
        # Tajima's D
        tajd_pg = diversity.tajimas_d(matrix)
        tajd_allel = allel.tajima_d(ac, pos=positions)
        print(f"\nTajima's D:")
        if np.isnan(tajd_pg):
            print(f"  pg_gpu:       nan")
        else:
            print(f"  pg_gpu:       {tajd_pg:.8f}")
        
        if np.isnan(tajd_allel):
            print(f"  scikit-allel: nan")
        else:
            print(f"  scikit-allel: {tajd_allel:.8f}")
        
        # Check NaN handling
        if np.isnan(tajd_pg) and np.isnan(tajd_allel):
            print(f"  Match:        True (both NaN)")
        else:
            print(f"  Match:        {np.isclose(tajd_pg, tajd_allel, rtol=1e-6)}")
    
    # Test with population subsets
    print(f"\n{'=' * 60}")
    print("Test: Population subsets")
    print('=' * 60)
    
    # Create data with two populations
    haplotypes = np.random.randint(0, 2, size=(40, 50), dtype=np.int8)
    # Add some differentiation
    haplotypes[:20, :25] = np.random.choice([0, 1], size=(20, 25), p=[0.8, 0.2])
    haplotypes[20:, :25] = np.random.choice([0, 1], size=(20, 25), p=[0.2, 0.8])
    
    positions = np.arange(50) * 1000 + 1000
    start_pos = positions[0]
    end_pos = positions[-1]
    
    matrix = HaplotypeMatrix(haplotypes, positions, start_pos, end_pos)
    matrix.sample_sets = {
        'pop1': list(range(20)),
        'pop2': list(range(20, 40))
    }
    
    print("\nComparing population-specific statistics:")
    
    for pop_name in ['pop1', 'pop2']:
        print(f"\nPopulation: {pop_name}")
        
        # Get population indices
        pop_indices = matrix.sample_sets[pop_name]
        pop_haplotypes = haplotypes[pop_indices, :]
        
        # pg_gpu with population
        pi_pg = diversity.pi(matrix, population=pop_name, span_normalize=True)
        
        # scikit-allel with population subset
        h_pop = allel.HaplotypeArray(pop_haplotypes.T)
        ac_pop = h_pop.count_alleles()
        pi_allel = allel.sequence_diversity(positions, ac_pop, start=start_pos, stop=end_pos)
        
        print(f"  Pi - pg_gpu:       {pi_pg:.8f}")
        print(f"  Pi - scikit-allel: {pi_allel:.8f}")
        print(f"  Match:             {np.isclose(pi_pg, pi_allel, rtol=1e-6)}")


if __name__ == "__main__":
    test_edge_cases()