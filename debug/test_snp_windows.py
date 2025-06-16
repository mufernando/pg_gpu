#!/usr/bin/env python
"""Debug SNP window iteration."""

import numpy as np
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import WindowParams, WindowIterator

# Test case from the test
n_variants = 100
positions = np.arange(n_variants) * 1000
haplotypes = np.random.randint(0, 2, size=(10, n_variants))

matrix = HaplotypeMatrix(haplotypes, positions)
params = WindowParams(window_type='snp', window_size=10, step_size=5)

iterator = WindowIterator(matrix, params)
windows = list(iterator)

print(f"Total windows: {len(windows)}")
print(f"Expected: 19")

# Show the start indices
for i, window in enumerate(windows):
    print(f"Window {i}: variants {window.matrix.positions[0]//1000:.0f}-{window.matrix.positions[-1]//1000:.0f} (n={window.n_variants})")
    
# Calculate manually
# Start indices: 0, 5, 10, 15, ..., 90
# That's (90 - 0) / 5 + 1 = 18 + 1 = 19 windows