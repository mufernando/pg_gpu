API Reference
=============

HaplotypeMatrix
---------------

.. autoclass:: pg_gpu.HaplotypeMatrix
   :members:
   :undoc-members:
   :show-inheritance:

Missing Data Methods
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pg_gpu.HaplotypeMatrix.has_missing
.. automethod:: pg_gpu.HaplotypeMatrix.is_missing
.. automethod:: pg_gpu.HaplotypeMatrix.is_called
.. automethod:: pg_gpu.HaplotypeMatrix.count_missing
.. automethod:: pg_gpu.HaplotypeMatrix.count_called
.. automethod:: pg_gpu.HaplotypeMatrix.get_span
.. automethod:: pg_gpu.HaplotypeMatrix.filter_variants_by_missing
.. automethod:: pg_gpu.HaplotypeMatrix.summarize_missing_data

LD Statistics Functions
-----------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd

.. autofunction:: pg_gpu.ld_statistics.dz

.. autofunction:: pg_gpu.ld_statistics.pi2

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd_within

.. autofunction:: pg_gpu.ld_statistics.dd_between

.. autofunction:: pg_gpu.ld_statistics.compute_ld_statistics

Parameters
----------

All LD statistics functions accept:

* **counts** : CuPy array of haplotype counts
* **populations** : Tuple specifying population indices (optional)
* **n_valid** : Valid sample counts for missing data (optional)

Population Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

* Single population: ``populations=None`` or omit
* Two populations: ``populations=(0, 1)`` for DD
* Three indices: ``populations=(0, 0, 1)`` for Dz
* Four indices: ``populations=(0, 0, 1, 1)`` for π₂

Diversity Statistics
--------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.diversity.pi
.. autofunction:: pg_gpu.diversity.theta_w
.. autofunction:: pg_gpu.diversity.tajimas_d
.. autofunction:: pg_gpu.diversity.haplotype_diversity
.. autofunction:: pg_gpu.diversity.allele_frequency_spectrum
.. autofunction:: pg_gpu.diversity.segregating_sites
.. autofunction:: pg_gpu.diversity.singleton_count
.. autofunction:: pg_gpu.diversity.fay_wus_h

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.diversity.diversity_stats
.. autofunction:: pg_gpu.diversity.neutrality_tests

Parameters
~~~~~~~~~~

All diversity functions support:

* **missing_data** : How to handle missing data ('include', 'exclude', 'ignore')
* **span_denominator** : For span normalization ('total', 'sites', 'callable')

Divergence Statistics
---------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.divergence.fst
.. autofunction:: pg_gpu.divergence.fst_hudson
.. autofunction:: pg_gpu.divergence.fst_weir_cockerham
.. autofunction:: pg_gpu.divergence.fst_nei
.. autofunction:: pg_gpu.divergence.dxy
.. autofunction:: pg_gpu.divergence.da

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.divergence.divergence_stats
.. autofunction:: pg_gpu.divergence.pairwise_fst

Parameters
~~~~~~~~~~

Divergence functions support:

* **missing_data** : How to handle missing data (where applicable)
* **span_denominator** : For span normalization (dxy, da, pi_within_population)