#!/usr/bin/env python
"""Compare polymorphism statistics between pg_gpu and scikit-allel."""

import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity

def compare_statistics():
    """Compare diversity statistics between pg_gpu and scikit-allel."""
    
    print("=" * 80)
    print("Comparing pg_gpu vs scikit-allel polymorphism statistics")
    print("=" * 80)
    
    # Create test datasets with different properties
    test_cases = [
        {
            "name": "Small dataset with known values",
            "haplotypes": np.array([
                [0, 0, 0, 1, 1],
                [0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=np.int8),
            "positions": np.array([100, 200, 300, 400, 500])
        },
        {
            "name": "Dataset with fixed sites",
            "haplotypes": np.array([
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 1, 1, 0, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 1, 0, 1]
            ], dtype=np.int8),
            "positions": np.array([1000, 2000, 3000, 4000, 5000])
        },
        {
            "name": "Larger random dataset",
            "haplotypes": np.random.randint(0, 2, size=(50, 100), dtype=np.int8),
            "positions": np.arange(100) * 1000
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest case: {test_case['name']}")
        print("-" * 60)
        
        haplotypes = test_case['haplotypes']
        positions = test_case['positions']
        n_haplotypes, n_variants = haplotypes.shape
        
        # Create pg_gpu HaplotypeMatrix
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        
        # Create scikit-allel HaplotypeArray
        h = allel.HaplotypeArray(haplotypes.T)  # scikit-allel uses transposed format
        
        # Compare allele counts
        print(f"\nDataset shape: {n_haplotypes} haplotypes, {n_variants} variants")
        
        # 1. Compare allele frequency spectrum
        print("\n1. Allele Frequency Spectrum:")
        
        # pg_gpu AFS
        afs_pg = diversity.allele_frequency_spectrum(matrix)
        
        # scikit-allel AFS
        ac = h.count_alleles()
        afs_allel = allel.sfs(ac[:, 1])  # SFS for derived allele
        
        print(f"   pg_gpu AFS:       {afs_pg.get() if hasattr(afs_pg, 'get') else afs_pg}")
        print(f"   scikit-allel SFS: {afs_allel}")
        
        # 2. Compare number of segregating sites
        print("\n2. Segregating Sites:")
        
        # pg_gpu
        seg_sites_pg = diversity.segregating_sites(matrix)
        
        # scikit-allel
        is_seg = ac.is_segregating()
        seg_sites_allel = np.sum(is_seg)
        
        print(f"   pg_gpu:       {seg_sites_pg}")
        print(f"   scikit-allel: {seg_sites_allel}")
        print(f"   Match: {seg_sites_pg == seg_sites_allel}")
        
        # 3. Compare pi (nucleotide diversity)
        print("\n3. Nucleotide Diversity (π):")
        
        # pg_gpu (without span normalization)
        pi_pg = diversity.pi(matrix, span_normalize=False)
        
        # scikit-allel
        pi_allel = allel.sequence_diversity(positions, ac)
        
        print(f"   pg_gpu:       {pi_pg:.6f}")
        print(f"   scikit-allel: {pi_allel:.6f}")
        print(f"   Difference:   {abs(pi_pg - pi_allel):.6f}")
        print(f"   Relative diff: {abs(pi_pg - pi_allel) / max(pi_pg, pi_allel) * 100:.2f}%")
        
        # 4. Compare Watterson's theta
        print("\n4. Watterson's Theta:")
        
        # pg_gpu (without span normalization)
        theta_pg = diversity.theta_w(matrix, span_normalize=False)
        
        # scikit-allel
        theta_allel = allel.watterson_theta(positions, ac)
        
        print(f"   pg_gpu:       {theta_pg:.6f}")
        print(f"   scikit-allel: {theta_allel:.6f}")
        print(f"   Difference:   {abs(theta_pg - theta_allel):.6f}")
        print(f"   Relative diff: {abs(theta_pg - theta_allel) / max(theta_pg, theta_allel) * 100:.2f}%")
        
        # 5. Compare Tajima's D
        print("\n5. Tajima's D:")
        
        # pg_gpu
        tajd_pg = diversity.tajimas_d(matrix)
        
        # scikit-allel
        tajd_allel = allel.tajima_d(ac, pos=positions)
        
        print(f"   pg_gpu:       {tajd_pg:.6f}")
        print(f"   scikit-allel: {tajd_allel:.6f}")
        print(f"   Difference:   {abs(tajd_pg - tajd_allel):.6f}")
        
        # 6. Compare singleton counts
        print("\n6. Singleton Count:")
        
        # pg_gpu
        singletons_pg = diversity.singleton_count(matrix)
        
        # scikit-allel
        is_singleton = ac.is_singleton(1)  # singletons for alt allele
        singletons_allel = np.sum(is_singleton)
        
        print(f"   pg_gpu:       {singletons_pg}")
        print(f"   scikit-allel: {singletons_allel}")
        print(f"   Match: {singletons_pg == singletons_allel}")
        
    print("\n" + "=" * 80)
    
    # Additional detailed comparison for debugging
    print("\nDetailed comparison for small dataset:")
    print("-" * 60)
    
    # Use first test case for detailed analysis
    haplotypes = test_cases[0]['haplotypes']
    positions = test_cases[0]['positions']
    
    print("\nHaplotype matrix:")
    print(haplotypes)
    
    print("\nAllele counts per site:")
    h = allel.HaplotypeArray(haplotypes.T)
    ac = h.count_alleles()
    for i, pos in enumerate(positions):
        ref_count = ac[i, 0]
        alt_count = ac[i, 1]
        print(f"   Position {pos}: ref={ref_count}, alt={alt_count}, freq={alt_count/(ref_count+alt_count):.3f}")


if __name__ == "__main__":
    compare_statistics()