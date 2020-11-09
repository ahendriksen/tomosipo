# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project *is planned to* adhere to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html) when it reaches version 1.0.

## [Unreleased]
### Added
- All geometry classes implement `len()`, which returns the number of steps in the geometry.
- Convenience methods `Transform.transform_vec` and `Transform.transform_point`
- `ts.svg()`: Render geometries to an animated SVG (no dependencies required(!)).
- `ts.svg()`: Added progress bar.
### Changed
- `ts.svg()`: Remove `base64` keyword parameter. This functionality has been
  moved to the `_repr_markdown_` method.
### Removed


## [0.2.0] - 2020-10-13
This is a major release. It introduces two features that define tomosipo in its current form:

1. Geometric transformations such as rotation, translation, and scaling are
   introduced. They can be composed and applied to all supported geometries.
2. Automatic linking of arrays, both for NumPy arrays and arrays on the GPU.
   This enables a paradigm shift in programming for ASTRA in Python, because it
   is no longer necessary to create `astra.data3d` objects. This greatly
   simplifies coding and makes the whole experience of using ASTRA considerably
   more pleasant.

### Added
- Support for parallel geometries (both parameterized and vector geometries)
- Volume vector geometry: this is a non-axis aligned version of the standard
  ASTRA volume geometry and replaces the Box object.
- PyTorch support
- CuPy support
- `tomosipo.qt.animate`: inline videos of geometries in Jupyter notebooks, which
  can also be saved to disk in `.mp4` format.
- Support for import ODL geometries
- `ts.operator`: this creates a wrapper for an ASTRA projector and can forward
  project arbitrary arrays. *Note:* This function might be renamed to
  `ts.projector`.
- `ts.to_astra` and `ts.from_astra`: generic conversion functions that convert
  geometry objects to and from their ASTRA representation.
### Changed
- Almost all APIs have been changed in some way or form. I will be more careful
  going forward.
- Moved from unittest to pytest framework for testing.

## [0.0.1] - 2018-09-25
### Added
- Initial support for cone and volume geometries
- Initial support for displaying data and geometries
- Conversion to and from astra geometry format



[Unreleased]: https://github.com/ahendriksen/tomosipo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ahendriksen/tomosipo/compare/v0.0.1...v0.2.0
[0.0.1]: https://github.com/ahendriksen/tomosipo/releases/tag/v0.0.1
