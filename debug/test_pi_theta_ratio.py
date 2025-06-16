#!/usr/bin/env python
"""Debug pi and theta_w ratio for random data."""

import numpy as np
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity

# Test with random data
n_variants = 1000
n_samples = 100

haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
positions = np.arange(n_variants) * 1000

matrix = HaplotypeMatrix(haplotypes, positions)

pi_value = diversity.pi(matrix, span_normalize=False)
theta_value = diversity.theta_w(matrix, span_normalize=False)

print(f"Pi value: {pi_value}")
print(f"Theta_w value: {theta_value}")
print(f"Ratio pi/theta_w: {pi_value / theta_value if theta_value > 0 else 'undefined'}")

# Also check the number of segregating sites
seg_sites = diversity.segregating_sites(matrix)
print(f"Segregating sites: {seg_sites}")
print(f"Total variants: {n_variants}")

# Expected values for random binary data
# Each site has probability 0.5^n_samples of being fixed
prob_segregating = 1 - 2 * (0.5 ** n_samples)
expected_seg_sites = n_variants * prob_segregating
print(f"\nExpected segregating sites (for random data): {expected_seg_sites:.2f}")

# For truly random data, the allele frequency at each site follows a binomial distribution
# This is NOT the same as neutral evolution!