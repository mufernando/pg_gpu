Changelog
=========

v0.3.0 (Current)
-----------------

New Modules
~~~~~~~~~~~

* **Selection Scan Statistics** (``pg_gpu.selection``)

  - ``ihs()`` - Integrated Haplotype Score with CUDA kernel acceleration
  - ``xpehh()`` - Cross-population Extended Haplotype Homozygosity
  - ``nsl()`` - Number of Segregating sites by Length
  - ``xpnsl()`` - Cross-population nSL
  - ``ehh_decay()`` - Extended Haplotype Homozygosity decay
  - ``garud_h()`` / ``moving_garud_h()`` - Garud's H1/H12/H123/H2H1 statistics
  - ``standardize()`` / ``standardize_by_allele_count()`` - Score normalization
  - 10-45x speedup over scikit-allel at typical dataset sizes

* **Site Frequency Spectrum** (``pg_gpu.sfs``)

  - ``sfs()`` / ``sfs_folded()`` / ``sfs_scaled()`` / ``sfs_folded_scaled()``
  - ``joint_sfs()`` / ``joint_sfs_folded()`` / ``joint_sfs_scaled()`` / ``joint_sfs_folded_scaled()``
  - ``fold_sfs()`` / ``fold_joint_sfs()`` - Folding utilities
  - Scaling utilities for neutral expectation comparison

* **Admixture / F-Statistics** (``pg_gpu.admixture``)

  - ``patterson_f2()`` - Branch length between populations
  - ``patterson_f3()`` - Three-population admixture test
  - ``patterson_d()`` - ABBA-BABA (D-statistic / F4)
  - ``moving_patterson_f3()`` / ``moving_patterson_d()`` - Windowed variants
  - ``average_patterson_f3()`` / ``average_patterson_d()`` - Block-jackknife with SE

New Functions
~~~~~~~~~~~~~

* **LD Statistics**

  - ``r()`` / ``r_squared()`` - Pearson correlation from haplotype counts
  - ``locate_unlinked()`` - GPU-accelerated LD pruning
  - ``windowed_r_squared()`` - Percentile of r-squared in distance bins

* **Diversity Statistics**

  - ``heterozygosity_expected()`` - Gene diversity per variant
  - ``heterozygosity_observed()`` - Observed heterozygosity (diploid)
  - ``inbreeding_coefficient()`` - Wright's F per variant

Infrastructure
~~~~~~~~~~~~~~

* Migrated to pixi for unified environment management
* Shared ``_utils.py`` module for population extraction
* Comprehensive validation test suite against scikit-allel

v0.2.0
------

New Features
~~~~~~~~~~~~

* **Comprehensive Missing Data Support**

  - All statistics support ``missing_data`` parameter ('include', 'exclude', 'ignore')
  - ``span_denominator`` parameter for flexible span normalization
  - HaplotypeMatrix methods for missing data analysis
  - Automatic detection and handling of missing data in LD statistics

* **New Statistics**

  - ``haplotype_diversity()`` - Haplotype diversity with Nei's correction
  - Population subsets in all diversity statistics

* **API Improvements**

  - Consistent parameter naming across all modules
  - Integration between diversity, divergence, and LD statistics

v0.1.0
------

Initial Release
~~~~~~~~~~~~~~~

* GPU-accelerated LD statistics (DD, Dz, pi2)
* Integration with moments package
* Basic missing data support in LD statistics
* Windowed analysis capabilities
