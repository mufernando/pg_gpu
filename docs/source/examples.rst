Examples
========

Complete Workflow
-----------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, ld_statistics, diversity, selection
   import numpy as np

   # Load VCF data
   h = HaplotypeMatrix.from_vcf("example.vcf")

   # Define populations
   h.sample_sets = {
       "CEU": [0, 1, 2, 3, 4],
       "YRI": [5, 6, 7, 8, 9]
   }

   # Diversity
   pi_ceu = diversity.pi(h, population="CEU")
   pi_yri = diversity.pi(h, population="YRI")

   # Divergence
   from pg_gpu import divergence
   fst = divergence.fst(h, "CEU", "YRI")

   # LD
   counts, n_valid = h.tally_gpu_haplotypes()
   r2 = ld_statistics.r_squared(counts, n_valid=n_valid)

   # Selection scans
   ihs_scores = selection.ihs(h, population="CEU")

Two-Population LD
-----------------

.. code-block:: python

   # Memory-efficient chunked computation
   stats = h.compute_ld_statistics_gpu_two_pops(
       bp_bins=np.array([0, 1000, 5000, 10000, 50000]),
       pop1="CEU",
       pop2="YRI",
       chunk_size='auto'
   )

Batch Statistics
----------------

.. code-block:: python

   # Multiple LD statistics in one call
   results = ld_statistics.compute_ld_statistics(
       counts,
       statistics=['dd', 'dz', 'pi2', 'r_squared'],
   )

Integration with moments
------------------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   h = HaplotypeMatrix.from_vcf("data.vcf")
   h.sample_sets = {"pop1": list(range(10))}

   # Compute LD statistics with GPU acceleration
   # chunk_size='auto' adapts to available GPU memory
   ld_stats = h.compute_ld_statistics_gpu_single_pop(
       bp_bins=[0, 1000, 5000],
       chunk_size='auto'
   )

LD Pruning
----------

.. code-block:: python

   # Find variants in linkage equilibrium
   unlinked = h.locate_unlinked(size=100, step=20, threshold=0.1)
   h_pruned = h.get_subset(np.where(unlinked)[0])
   print(f"Kept {np.sum(unlinked)} of {h.num_variants} variants")

   # Windowed r-squared decay
   bins = np.arange(0, 100001, 1000)
   r2_decay, counts = h.windowed_r_squared(bins, percentile=50)

Selection Scan Pipeline
-----------------------

.. code-block:: python

   from pg_gpu import selection

   # iHS with standardization by allele count
   ihs_raw = selection.ihs(h)

   # Get allele counts for binned standardization
   dac = np.sum(h.haplotypes.get(), axis=0)
   ihs_std, bins = selection.standardize_by_allele_count(ihs_raw, dac)

   # Cross-population scans
   xpehh_scores = selection.xpehh(h, "CEU", "YRI")
   xpehh_std = selection.standardize(xpehh_scores)

   # Garud's H in sliding windows
   h1, h12, h123, h2_h1 = selection.moving_garud_h(h, size=200, step=50)

SFS and Admixture
-----------------

.. code-block:: python

   from pg_gpu import sfs, admixture

   # Joint SFS
   jsfs = sfs.joint_sfs(h, "CEU", "YRI")

   # Patterson's D with block-jackknife
   d, se, z, vb, vj = admixture.average_patterson_d(
       h, "popA", "popB", "popC", "popD", blen=100
   )
   print(f"D = {d:.4f}, SE = {se:.4f}, Z = {z:.2f}")

Missing Data
------------

.. code-block:: python

   from pg_gpu import diversity

   # Check missing data summary
   summary = h.summarize_missing_data()
   print(f"Missing: {summary['missing_freq_overall']:.1%}")

   # Different strategies
   pi_include = diversity.pi(h, missing_data='include')
   pi_exclude = diversity.pi(h, missing_data='exclude')

   # LD statistics handle missing data automatically
   counts, n_valid = h.tally_gpu_haplotypes()
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)
