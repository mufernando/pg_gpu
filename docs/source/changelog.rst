Changelog
=========

v0.2.0 (Current)
----------------

New Features
~~~~~~~~~~~~

* **Comprehensive Missing Data Support**
  
  - All statistics now support ``missing_data`` parameter ('include', 'exclude', 'ignore')
  - Added ``span_denominator`` parameter for flexible span normalization
  - New HaplotypeMatrix methods for missing data analysis
  - Automatic detection and handling of missing data in LD statistics

* **New Statistics**
  
  - ``haplotype_diversity()`` - Compute haplotype diversity with Nei's correction
  - Support for population subsets in all diversity statistics

* **API Improvements**
  
  - Consistent parameter naming across all modules
  - Better integration between diversity, divergence, and LD statistics
  - Enhanced documentation with missing data examples

Breaking Changes
~~~~~~~~~~~~~~~~

* Diversity and divergence functions now require explicit ``missing_data`` parameter when handling missing data (default='include')
* ``span_normalize`` parameter now works with ``span_denominator`` for more control

Bug Fixes
~~~~~~~~~

* Fixed span normalization in windowed analysis
* Corrected negative probability issues in test data generation
* Improved numerical precision in haplotype diversity calculations

v0.1.0
------

Initial Release
~~~~~~~~~~~~~~~

* GPU-accelerated LD statistics (DD, Dz, π₂)
* Integration with moments package
* Basic missing data support in LD statistics
* Windowed analysis capabilities