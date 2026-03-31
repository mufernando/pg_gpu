"""
Validation tests comparing pg_gpu SFS functions against scikit-allel.
"""

import pytest
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import sfs


@pytest.fixture
def single_pop_data():
    """Deterministic single-population haplotype data."""
    np.random.seed(42)
    n_hap = 20
    n_var = 100
    hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
    pos = np.arange(n_var) * 1000
    matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)
    return matrix, hap


@pytest.fixture
def two_pop_data():
    """Deterministic two-population data."""
    np.random.seed(123)
    n_var = 100
    n1, n2 = 10, 12
    hap = np.random.randint(0, 2, (n1 + n2, n_var), dtype=np.int8)
    pos = np.arange(n_var) * 1000
    matrix = HaplotypeMatrix(
        hap, pos, 0, n_var * 1000,
        sample_sets={'pop1': list(range(n1)), 'pop2': list(range(n1, n1 + n2))}
    )
    return matrix, hap, n1, n2


class TestSFSComparison:
    """Validate SFS against scikit-allel."""

    def test_sfs(self, single_pop_data):
        matrix, hap = single_pop_data
        n = hap.shape[0]
        # pg_gpu
        result = sfs.sfs(matrix)
        # allel: pass n explicitly to match full-size output
        dac = np.sum(hap, axis=0)
        expected = allel.sfs(dac, n=n)
        np.testing.assert_array_equal(result, expected)

    def test_sfs_folded(self, single_pop_data):
        matrix, hap = single_pop_data
        result = sfs.sfs_folded(matrix)
        # allel: needs allele counts (n_variants, 2)
        dac = np.sum(hap, axis=0)
        n = hap.shape[0]
        ac = np.column_stack([n - dac, dac])
        expected = allel.sfs_folded(ac)
        np.testing.assert_array_equal(result, expected)

    def test_sfs_scaled(self, single_pop_data):
        matrix, hap = single_pop_data
        n = hap.shape[0]
        result = sfs.sfs_scaled(matrix)
        dac = np.sum(hap, axis=0)
        expected = allel.sfs_scaled(dac, n=n)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sfs_folded_scaled(self, single_pop_data):
        matrix, hap = single_pop_data
        result = sfs.sfs_folded_scaled(matrix)
        dac = np.sum(hap, axis=0)
        n = hap.shape[0]
        ac = np.column_stack([n - dac, dac])
        expected = allel.sfs_folded_scaled(ac)
        np.testing.assert_array_almost_equal(result, expected)


class TestJointSFSComparison:
    """Validate joint SFS against scikit-allel."""

    def test_joint_sfs(self, two_pop_data):
        matrix, hap, n1, n2 = two_pop_data
        result = sfs.joint_sfs(matrix, 'pop1', 'pop2')
        dac1 = np.sum(hap[:n1], axis=0)
        dac2 = np.sum(hap[n1:], axis=0)
        expected = allel.joint_sfs(dac1, dac2, n1=n1, n2=n2)
        np.testing.assert_array_equal(result, expected)

    def test_joint_sfs_folded(self, two_pop_data):
        matrix, hap, n1, n2 = two_pop_data
        result = sfs.joint_sfs_folded(matrix, 'pop1', 'pop2')
        dac1 = np.sum(hap[:n1], axis=0)
        dac2 = np.sum(hap[n1:], axis=0)
        ac1 = np.column_stack([n1 - dac1, dac1])
        ac2 = np.column_stack([n2 - dac2, dac2])
        expected = allel.joint_sfs_folded(ac1, ac2)
        np.testing.assert_array_equal(result, expected)

    def test_joint_sfs_scaled(self, two_pop_data):
        matrix, hap, n1, n2 = two_pop_data
        result = sfs.joint_sfs_scaled(matrix, 'pop1', 'pop2')
        dac1 = np.sum(hap[:n1], axis=0)
        dac2 = np.sum(hap[n1:], axis=0)
        expected = allel.joint_sfs_scaled(dac1, dac2, n1=n1, n2=n2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_joint_sfs_folded_scaled(self, two_pop_data):
        matrix, hap, n1, n2 = two_pop_data
        result = sfs.joint_sfs_folded_scaled(matrix, 'pop1', 'pop2')
        dac1 = np.sum(hap[:n1], axis=0)
        dac2 = np.sum(hap[n1:], axis=0)
        ac1 = np.column_stack([n1 - dac1, dac1])
        ac2 = np.column_stack([n2 - dac2, dac2])
        expected = allel.joint_sfs_folded_scaled(ac1, ac2)
        np.testing.assert_array_almost_equal(result, expected)


class TestScalingComparison:
    """Validate scaling and folding utilities."""

    def test_scale_sfs(self):
        s = np.array([10, 5, 3, 2, 1])
        np.testing.assert_array_equal(
            sfs.scale_sfs(s), allel.scale_sfs(s))

    def test_scale_sfs_folded(self):
        s = np.array([10, 5, 3])
        np.testing.assert_array_almost_equal(
            sfs.scale_sfs_folded(s, 4), allel.scale_sfs_folded(s, 4))

    def test_scale_joint_sfs(self):
        s = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(
            sfs.scale_joint_sfs(s), allel.scale_joint_sfs(s))

    def test_fold_sfs(self):
        for n in [8, 10, 20]:
            np.random.seed(42)
            s = np.random.randint(0, 20, size=n + 1)
            np.testing.assert_array_equal(
                sfs.fold_sfs(s, n), allel.fold_sfs(s, n))
