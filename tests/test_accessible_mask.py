"""Tests for the accessible site mask feature."""

import numpy as np
import os
import tempfile
import pytest

from pg_gpu.accessible import AccessibleMask, parse_bed, bed_to_mask
from pg_gpu.haplotype_matrix import HaplotypeMatrix
from pg_gpu.genotype_matrix import GenotypeMatrix


# ---- AccessibleMask unit tests ----

class TestAccessibleMask:
    def test_basic_construction(self):
        mask = np.array([True, True, False, True, False], dtype=bool)
        am = AccessibleMask(mask, offset=100)
        assert len(am) == 5
        assert am.offset == 100
        assert am.total_accessible == 3

    def test_count_accessible_full_range(self):
        mask = np.array([True, True, False, True, False], dtype=bool)
        am = AccessibleMask(mask, offset=100)
        assert am.count_accessible(100, 105) == 3

    def test_count_accessible_subrange(self):
        mask = np.array([True, True, False, True, False], dtype=bool)
        am = AccessibleMask(mask, offset=100)
        assert am.count_accessible(100, 102) == 2
        assert am.count_accessible(102, 104) == 1
        assert am.count_accessible(103, 105) == 1

    def test_count_accessible_out_of_bounds(self):
        mask = np.array([True, True, False], dtype=bool)
        am = AccessibleMask(mask, offset=100)
        # Partially out of range
        assert am.count_accessible(99, 102) == 2
        # Completely out of range
        assert am.count_accessible(200, 300) == 0
        # Reversed range
        assert am.count_accessible(105, 100) == 0

    def test_count_accessible_prefix_sum_consistency(self):
        rng = np.random.RandomState(42)
        mask = rng.choice([True, False], size=1000)
        am = AccessibleMask(mask, offset=500)
        # Verify prefix sum matches naive count for random ranges
        for _ in range(50):
            s = rng.randint(500, 1400)
            e = rng.randint(s, 1600)
            expected = np.count_nonzero(
                mask[max(0, s - 500):min(1000, e - 500)])
            assert am.count_accessible(s, e) == expected

    def test_slice_basic(self):
        mask = np.array([True, True, False, True, False, True], dtype=bool)
        am = AccessibleMask(mask, offset=100)
        sliced = am.slice(102, 105)
        assert sliced.offset == 102
        assert len(sliced) == 3
        np.testing.assert_array_equal(
            sliced.mask, [False, True, False])

    def test_slice_out_of_bounds(self):
        mask = np.array([True, True, False], dtype=bool)
        am = AccessibleMask(mask, offset=100)
        sliced = am.slice(200, 300)
        assert len(sliced) == 0
        assert sliced.total_accessible == 0

    def test_repr(self):
        mask = np.ones(100, dtype=bool)
        am = AccessibleMask(mask, offset=0)
        r = repr(am)
        assert "AccessibleMask" in r
        assert "100" in r

    def test_zero_offset(self):
        mask = np.array([True, False, True], dtype=bool)
        am = AccessibleMask(mask)
        assert am.offset == 0
        assert am.count_accessible(0, 3) == 2


# ---- BED parsing tests ----

class TestParseBed:
    def _write_bed(self, content):
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.bed', delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_basic_parsing(self):
        path = self._write_bed("chr1\t100\t200\nchr1\t300\t400\n")
        try:
            intervals = parse_bed(path)
            assert intervals == [("chr1", 100, 200), ("chr1", 300, 400)]
        finally:
            os.unlink(path)

    def test_chrom_filter(self):
        path = self._write_bed(
            "chr1\t100\t200\nchr2\t300\t400\nchr1\t500\t600\n")
        try:
            intervals = parse_bed(path, chrom="chr1")
            assert len(intervals) == 2
            assert all(c == "chr1" for c, _, _ in intervals)
        finally:
            os.unlink(path)

    def test_skip_comments_and_headers(self):
        path = self._write_bed(
            "# comment\ntrack name=foo\nbrowser position\n"
            "chr1\t100\t200\n")
        try:
            intervals = parse_bed(path)
            assert len(intervals) == 1
            assert intervals[0] == ("chr1", 100, 200)
        finally:
            os.unlink(path)

    def test_extra_columns(self):
        path = self._write_bed("chr1\t100\t200\tname\t0\t+\n")
        try:
            intervals = parse_bed(path)
            assert intervals == [("chr1", 100, 200)]
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = self._write_bed("")
        try:
            intervals = parse_bed(path)
            assert intervals == []
        finally:
            os.unlink(path)

    def test_space_delimited(self):
        path = self._write_bed("chr1 100 200\n")
        try:
            intervals = parse_bed(path)
            assert intervals == [("chr1", 100, 200)]
        finally:
            os.unlink(path)


class TestBedToMask:
    def _write_bed(self, content):
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.bed', delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_basic_mask(self):
        path = self._write_bed("chr1\t10\t20\nchr1\t30\t35\n")
        try:
            am = bed_to_mask(path, chrom="chr1", length=50, offset=0)
            assert am.total_accessible == 15  # 10 + 5
            assert am.count_accessible(10, 20) == 10
            assert am.count_accessible(0, 10) == 0
            assert am.count_accessible(30, 35) == 5
        finally:
            os.unlink(path)

    def test_mask_with_offset(self):
        path = self._write_bed("chr1\t100\t200\n")
        try:
            am = bed_to_mask(path, chrom="chr1", length=200, offset=50)
            # Mask covers positions 50-250, BED interval is 100-200
            # Mask indices: 50->0, 100->50, 200->150
            assert am.count_accessible(100, 200) == 100
            assert am.count_accessible(50, 100) == 0
        finally:
            os.unlink(path)

    def test_overlapping_intervals(self):
        path = self._write_bed("chr1\t10\t20\nchr1\t15\t25\n")
        try:
            am = bed_to_mask(path, chrom="chr1", length=30, offset=0)
            # Overlapping region 15-20 should be True once
            assert am.total_accessible == 15  # 10-25
        finally:
            os.unlink(path)


# ---- HaplotypeMatrix integration tests ----

def _make_haplotype_matrix(n_hap=10, n_var=20, chrom_start=0,
                           chrom_end=1000):
    """Helper to create a simple HaplotypeMatrix."""
    rng = np.random.RandomState(42)
    hap = rng.randint(0, 2, size=(n_hap, n_var)).astype(np.int8)
    pos = np.sort(rng.choice(
        np.arange(chrom_start + 1, chrom_end), n_var, replace=False))
    return HaplotypeMatrix(hap, pos, chrom_start=chrom_start,
                           chrom_end=chrom_end)


class TestHaplotypeMatrixAccessibleMask:
    def test_init_with_mask_array(self):
        mask = np.ones(1000, dtype=bool)
        mask[500:600] = False
        hm = _make_haplotype_matrix()
        hm_with_mask = HaplotypeMatrix(
            hm.haplotypes, hm.positions,
            chrom_start=0, chrom_end=1000,
            accessible_mask=mask)
        assert hm_with_mask.has_accessible_mask
        assert hm_with_mask.accessible_mask.total_accessible == 900
        # n_total_sites should be derived from mask
        assert hm_with_mask.n_total_sites == 900

    def test_init_with_accessible_mask_object(self):
        am = AccessibleMask(np.ones(500, dtype=bool), offset=100)
        hm = _make_haplotype_matrix(chrom_start=100, chrom_end=600)
        hm2 = HaplotypeMatrix(
            hm.haplotypes, hm.positions,
            chrom_start=100, chrom_end=600,
            accessible_mask=am)
        assert hm2.has_accessible_mask
        assert hm2.accessible_mask is am

    def test_init_no_mask(self):
        hm = _make_haplotype_matrix()
        assert not hm.has_accessible_mask
        assert hm.accessible_mask is None

    def test_set_accessible_mask_array(self):
        hm = _make_haplotype_matrix()
        mask = np.ones(1000, dtype=bool)
        mask[200:300] = False
        hm.set_accessible_mask(mask)
        assert hm.has_accessible_mask
        assert hm.accessible_mask.total_accessible == 900
        assert hm.accessible_mask.offset == 0

    def test_set_accessible_mask_bed(self):
        hm = _make_haplotype_matrix(chrom_start=0, chrom_end=1000)
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.bed', delete=False)
        f.write("chr1\t0\t500\nchr1\t600\t1000\n")
        f.close()
        try:
            hm.set_accessible_mask(f.name, chrom="chr1")
            assert hm.has_accessible_mask
            assert hm.accessible_mask.total_accessible == 900
        finally:
            os.unlink(f.name)

    def test_set_accessible_mask_object(self):
        hm = _make_haplotype_matrix()
        am = AccessibleMask(np.ones(1000, dtype=bool), offset=0)
        hm.set_accessible_mask(am)
        assert hm.accessible_mask is am

    def test_n_total_sites_not_overwritten(self):
        hm = _make_haplotype_matrix()
        hm.n_total_sites = 5000
        mask = np.ones(1000, dtype=bool)
        hm2 = HaplotypeMatrix(
            hm.haplotypes, hm.positions,
            chrom_start=0, chrom_end=1000,
            n_total_sites=5000, accessible_mask=mask)
        # When n_total_sites is explicitly set, it should not be overridden
        assert hm2.n_total_sites == 5000

    def test_get_span_accessible(self):
        mask = np.ones(1000, dtype=bool)
        mask[200:300] = False  # 100 inaccessible
        hm = _make_haplotype_matrix()
        hm.set_accessible_mask(mask)
        assert hm.get_span('accessible') == 900
        assert hm.get_span('total') == 1000

    def test_get_span_accessible_no_mask_fallback(self):
        hm = _make_haplotype_matrix()
        # Without mask, 'accessible' should fall back to 'total'
        span = hm.get_span('accessible')
        assert span == hm.get_span('total')

    def test_get_subset_propagates_mask(self):
        mask = np.ones(1000, dtype=bool)
        hm = _make_haplotype_matrix()
        hm.set_accessible_mask(mask)
        indices = np.array([0, 1, 2])
        subset = hm.get_subset(indices)
        assert subset.has_accessible_mask
        assert subset.accessible_mask is hm.accessible_mask

    def test_get_subset_propagates_n_total_sites(self):
        hm = _make_haplotype_matrix()
        hm.n_total_sites = 5000
        indices = np.array([0, 1, 2])
        subset = hm.get_subset(indices)
        assert subset.n_total_sites == 5000

    def test_get_subset_empty_propagates(self):
        mask = np.ones(1000, dtype=bool)
        hm = _make_haplotype_matrix()
        hm.set_accessible_mask(mask)
        hm.n_total_sites = 999
        indices = np.array([], dtype=np.int64)
        subset = hm.get_subset(indices)
        assert subset.has_accessible_mask
        assert subset.n_total_sites == 999

    def test_get_subset_from_range_slices_mask(self):
        # Use a matrix with enough variants to allow range subsetting
        rng = np.random.RandomState(123)
        n_var = 100
        hap = rng.randint(0, 2, size=(10, n_var)).astype(np.int8)
        pos = np.sort(rng.choice(np.arange(10, 990), n_var, replace=False))
        hm = HaplotypeMatrix(hap, pos, chrom_start=0, chrom_end=1000)
        mask = np.ones(1000, dtype=bool)
        mask[40:60] = False  # 20 inaccessible
        hm.set_accessible_mask(mask)
        # get_subset_from_range checks high <= positions.size
        # so use a range within the number of positions
        low, high = 30, 80
        subset = hm.get_subset_from_range(low, high)
        assert subset.has_accessible_mask
        assert subset.accessible_mask.offset == low
        assert len(subset.accessible_mask) == (high - low)
        # 40-60 is inaccessible; within [30,80) that's 20 inaccessible
        assert subset.accessible_mask.total_accessible == 30


# ---- GenotypeMatrix integration tests ----

class TestGenotypeMatrixAccessibleMask:
    def test_from_haplotype_matrix_propagates(self):
        mask = np.ones(1000, dtype=bool)
        hm = _make_haplotype_matrix(n_hap=10)
        hm.set_accessible_mask(mask)
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        assert gm.has_accessible_mask
        assert gm.accessible_mask is hm.accessible_mask

    def test_to_haplotype_matrix_propagates(self):
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 3, size=(5, 20)).astype(np.int8)
        pos = np.sort(rng.choice(np.arange(1, 1000), 20, replace=False))
        mask = np.ones(1000, dtype=bool)
        am = AccessibleMask(mask, offset=0)
        gm = GenotypeMatrix(geno, pos, chrom_start=0, chrom_end=1000,
                            accessible_mask=am)
        hm = gm.to_haplotype_matrix()
        assert hm.has_accessible_mask
        assert hm.accessible_mask is am

    def test_apply_biallelic_filter_propagates(self):
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 3, size=(5, 20)).astype(np.int8)
        pos = np.sort(rng.choice(np.arange(1, 1000), 20, replace=False))
        mask = np.ones(1000, dtype=bool)
        am = AccessibleMask(mask, offset=0)
        gm = GenotypeMatrix(geno, pos, chrom_start=0, chrom_end=1000,
                            accessible_mask=am)
        filtered = gm.apply_biallelic_filter()
        assert filtered.has_accessible_mask
        assert filtered.accessible_mask is am


# ---- Population matrix propagation ----

class TestPopulationMatrixPropagation:
    def test_utils_get_population_matrix(self):
        from pg_gpu._utils import get_population_matrix
        hm = _make_haplotype_matrix(n_hap=10)
        hm._sample_sets = {'pop1': [0, 1, 2], 'pop2': [3, 4, 5]}
        mask = np.ones(1000, dtype=bool)
        hm.set_accessible_mask(mask)
        pop_hm = get_population_matrix(hm, 'pop1')
        assert pop_hm.has_accessible_mask
        assert pop_hm.accessible_mask is hm.accessible_mask
        assert pop_hm.n_total_sites == hm.n_total_sites


# ---- Backward compatibility ----

class TestBackwardCompatibility:
    def test_no_mask_operations(self):
        """All operations work without a mask set."""
        hm = _make_haplotype_matrix()
        assert not hm.has_accessible_mask
        # get_span should still work
        assert hm.get_span('total') == 1000
        assert hm.get_span('sites') == 20
        # get_subset should work
        subset = hm.get_subset(np.array([0, 1]))
        assert not subset.has_accessible_mask

    def test_accessible_fallback_to_total(self):
        hm = _make_haplotype_matrix()
        assert hm.get_span('accessible') == hm.get_span('total')

    def test_apply_biallelic_filter_propagates_mask(self):
        """HaplotypeMatrix.apply_biallelic_filter preserves mask."""
        hm = _make_haplotype_matrix()
        mask = np.ones(1000, dtype=bool)
        am = AccessibleMask(mask, offset=0)
        hm.set_accessible_mask(am)
        hm.n_total_sites = 5000
        filtered = hm.apply_biallelic_filter()
        assert filtered.has_accessible_mask
        assert filtered.accessible_mask is am
        assert filtered.n_total_sites == 5000


# ---- filter_to_accessible tests ----

class TestFilterToAccessible:
    def test_no_mask_returns_self(self):
        hm = _make_haplotype_matrix()
        assert hm.filter_to_accessible() is hm

    def test_all_accessible_returns_self(self):
        hm = _make_haplotype_matrix()
        mask = np.ones(1000, dtype=bool)
        hm.set_accessible_mask(mask)
        assert hm.filter_to_accessible() is hm

    def test_set_accessible_mask_filters_immediately(self):
        """set_accessible_mask filters variants at inaccessible positions."""
        rng = np.random.RandomState(42)
        n_var = 50
        hap = rng.randint(0, 2, size=(10, n_var)).astype(np.int8)
        pos = np.arange(100, 100 + n_var * 10, 10)  # 100, 110, ..., 590
        hm = HaplotypeMatrix(hap, pos, chrom_start=0, chrom_end=600)
        original_nvar = hm.num_variants
        # Mark positions 200-400 as inaccessible
        mask = np.ones(600, dtype=bool)
        mask[200:400] = False
        hm.set_accessible_mask(mask)

        # Variants should be filtered immediately
        assert hm.num_variants < original_nvar
        for p in hm.positions:
            assert int(p) < 200 or int(p) >= 400
        # Mask retained for span normalization
        assert hm.has_accessible_mask

    def test_filter_to_accessible_noop_after_set(self):
        """filter_to_accessible is a no-op after set_accessible_mask."""
        rng = np.random.RandomState(42)
        hap = rng.randint(0, 2, size=(10, 50)).astype(np.int8)
        pos = np.arange(100, 600, 10)
        hm = HaplotypeMatrix(hap, pos, chrom_start=0, chrom_end=600)
        mask = np.ones(600, dtype=bool)
        mask[200:400] = False
        hm.set_accessible_mask(mask)
        # filter_to_accessible should return self (already filtered)
        assert hm.filter_to_accessible() is hm

    def test_genotype_matrix_set_mask_filters(self):
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 3, size=(5, 30)).astype(np.int8)
        pos = np.arange(10, 310, 10)
        gm = GenotypeMatrix(geno, pos, chrom_start=0, chrom_end=320)
        original_nvar = gm.num_variants
        mask = np.ones(320, dtype=bool)
        mask[100:200] = False
        gm.set_accessible_mask(mask)
        assert gm.num_variants < original_nvar
        for p in gm.positions:
            assert int(p) < 100 or int(p) >= 200


# ---- Windowed analysis integration tests ----

def _make_windowed_matrix():
    """Create a HaplotypeMatrix suitable for windowed analysis."""
    rng = np.random.RandomState(42)
    n_hap, n_var = 20, 200
    hap = rng.randint(0, 2, size=(n_hap, n_var)).astype(np.int8)
    pos = np.linspace(100, 9900, n_var).astype(int)
    hm = HaplotypeMatrix(hap, pos, chrom_start=0, chrom_end=10000)
    return hm


class TestWindowedFusedWithMask:
    def test_fused_per_base_uses_mask(self):
        """Windowed fused path normalizes by accessible bases, not span."""
        from pg_gpu.windowed_analysis import windowed_statistics_fused

        hm = _make_windowed_matrix()
        hm.transfer_to_gpu()

        # Mask: only first half accessible
        mask = np.zeros(10000, dtype=bool)
        mask[0:5000] = True
        hm.set_accessible_mask(mask)

        bp_bins = np.array([0, 5000, 10000], dtype=float)
        result = windowed_statistics_fused(
            hm, bp_bins, statistics=('pi',), per_base=True)

        # Window [0, 5000): 5000 accessible bases
        # Window [5000, 10000): 0 accessible bases -> NaN
        assert not np.isnan(result['pi'][0])
        assert np.isnan(result['pi'][1])

    def test_fused_explicit_mask_overrides_attribute(self):
        """Explicit is_accessible param takes precedence over matrix attribute."""
        from pg_gpu.windowed_analysis import windowed_statistics_fused

        hm = _make_windowed_matrix()
        hm.transfer_to_gpu()

        # Matrix mask: all accessible
        mask_all = np.ones(10000, dtype=bool)
        hm.set_accessible_mask(mask_all)

        # Explicit mask: nothing accessible
        mask_none = np.zeros(10000, dtype=bool)
        bp_bins = np.array([0, 10000], dtype=float)
        result = windowed_statistics_fused(
            hm, bp_bins, statistics=('pi',), per_base=True,
            is_accessible=mask_none)

        # Explicit mask should win -> NaN
        assert np.isnan(result['pi'][0])


class TestWindowedScatterAddWithMask:
    def test_scatter_add_per_base_uses_mask(self):
        """Windowed scatter-add path normalizes by accessible bases."""
        from pg_gpu.windowed_analysis import windowed_statistics

        hm = _make_windowed_matrix()
        hm.transfer_to_gpu()

        mask = np.zeros(10000, dtype=bool)
        mask[0:5000] = True
        hm.set_accessible_mask(mask)

        bp_bins = np.array([0, 5000, 10000], dtype=float)
        result = windowed_statistics(
            hm, bp_bins, statistics=('pi',), per_base=True)

        assert not np.isnan(result['pi'][0])
        assert np.isnan(result['pi'][1])


class TestWindowedAnalysisConvenience:
    def test_accessible_span_denominator(self):
        """windowed_analysis with span_denominator='accessible' uses mask."""
        from pg_gpu.windowed_analysis import windowed_analysis

        hm = _make_windowed_matrix()

        mask = np.ones(10000, dtype=bool)
        mask[2000:3000] = False  # 1000 inaccessible in some windows
        hm.set_accessible_mask(mask)

        df = windowed_analysis(
            hm, window_size=5000, statistics=['pi'],
            span_denominator='accessible')

        assert len(df) > 0
        assert 'pi' in df.columns

    def test_accessible_bed_parameter(self):
        """windowed_analysis loads BED mask when accessible_bed provided."""
        from pg_gpu.windowed_analysis import windowed_analysis

        hm = _make_windowed_matrix()

        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.bed', delete=False)
        f.write("chr1\t0\t10000\n")
        f.close()
        try:
            df = windowed_analysis(
                hm, window_size=5000, statistics=['pi'],
                accessible_bed=f.name)
            assert hm.has_accessible_mask
            assert len(df) > 0
        finally:
            os.unlink(f.name)


# ---- Pairwise mode + accessible mask ----

class TestPairwiseModeWithMask:
    def test_n_total_sites_derived_per_window(self):
        """In pairwise windowed mode, n_total_sites is per-window accessible."""
        from pg_gpu.windowed_analysis import WindowIterator, WindowParams

        hm = _make_windowed_matrix()
        mask = np.ones(10000, dtype=bool)
        mask[5000:6000] = False  # 1000 inaccessible
        hm.set_accessible_mask(mask)

        params = WindowParams(window_type='bp', window_size=5000,
                              step_size=5000)
        iterator = WindowIterator(hm, params)

        windows = list(iterator)
        assert len(windows) >= 2

        # First window [100, 5100): fully accessible
        w1 = windows[0]
        assert w1.matrix.has_accessible_mask

        # Second window should have fewer accessible sites due to mask
        w2 = windows[1]
        assert w2.matrix.has_accessible_mask
        assert w2.matrix.n_total_sites < w1.matrix.n_total_sites


# ---- Selection module integration ----

class TestSelectionAccessibleMask:
    def test_ihs_uses_matrix_mask(self):
        """ihs() falls back to matrix's accessible mask."""
        from pg_gpu import selection

        rng = np.random.RandomState(42)
        n_hap, n_var = 40, 100
        hap = rng.randint(0, 2, size=(n_hap, n_var)).astype(np.int8)
        pos = np.linspace(1, 100000, n_var).astype(int)
        hm = HaplotypeMatrix(hap, pos, chrom_start=0,
                             chrom_end=100001)

        # Without mask
        score_no_mask = selection.ihs(hm)

        # With fully accessible mask (should give same result)
        mask = np.ones(100001, dtype=bool)
        hm.set_accessible_mask(mask)
        score_with_mask = selection.ihs(hm)

        np.testing.assert_array_equal(score_no_mask, score_with_mask)
