"""
Shared utilities for pg_gpu modules.
"""

from typing import Union
from .haplotype_matrix import HaplotypeMatrix


def get_population_matrix(haplotype_matrix: HaplotypeMatrix,
                          population: Union[str, list]) -> HaplotypeMatrix:
    """Extract a population-specific HaplotypeMatrix.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The full haplotype data.
    population : str or list
        Population name (looked up in sample_sets) or list of sample indices.

    Returns
    -------
    HaplotypeMatrix
        Subset matrix for the specified population.
    """
    if isinstance(population, str):
        if haplotype_matrix.sample_sets is None:
            raise ValueError("No sample_sets defined in haplotype matrix")
        if population not in haplotype_matrix.sample_sets:
            raise ValueError(
                f"Population {population} not found in sample_sets")
        pop_indices = haplotype_matrix.sample_sets[population]
    else:
        pop_indices = list(population)

    pop_haplotypes = haplotype_matrix.haplotypes[pop_indices, :]
    return HaplotypeMatrix(
        pop_haplotypes,
        haplotype_matrix.positions,
        haplotype_matrix.chrom_start,
        haplotype_matrix.chrom_end,
        sample_sets={'all': list(range(len(pop_indices)))},
        n_total_sites=haplotype_matrix.n_total_sites,
    )
