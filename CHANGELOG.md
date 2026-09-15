# Changelog

## 0.9.0 - 2026-09-15

### Added

- Added the `fit_tscan_qt` PyQt5 workflow for data loading, model configuration,
  fitting, synchronized result and residual plots, export, and CI scans.
- Added 1-sigma and 2-sigma parameter uncertainty estimation through CI scans.
- Added the `calc_dads_qt` PyQt5 workflow for DADS, SADS, SVD-assisted analysis,
  custom real-valued rate models, visualization, and export bundles.
- Added headless PyQt5 workflow tests and a GitHub Actions test matrix for Python
  3.11, 3.12, and 3.13.

### Changed

- Declared Python 3.11 as the minimum supported Python version.
- Aligned the package NumPy requirement with the runtime and test requirements.
- Updated installation and source-tree launch documentation for both Qt apps.

### Fixed

- Kept Matplotlib on the PyQt5 backend in the Qt applications and tests.
- Made NumPy-based tests deterministic across local runs and CI shards.
- Removed invalid-escape warnings from the legacy Tk GUI input validators.
- Corrected the CalcDADS Qt wrapper description.

### Scope

- Custom SADS models in `calc_dads_qt` are intentionally limited to real-valued
  first-order rate models. Complex eigenmodes and oscillatory terms are not part
  of the Qt workflow in this release.
