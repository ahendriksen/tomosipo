# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project *is planned to* adhere to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html) when it reaches version 1.0.

## [Unreleased] -
### Added
### Changed
### Fixed
### Removed
### Deprecated
### Security

## [0.6.0] - 2022-07-19
### Added
- References and links to ts_algorithms were added to the readme
### Changed
- Changed the Tomosipo autograd operators made by the tomosipo.torch\_support.to\_autograd function. By default they now expect the same input shape (3 dimensions) as a normal Tomosipo operator. New arguments have been added to also support the input shapes expected in Pytorch neural networks.

## [0.5.1] - 2022-06-15
Small bugfix to the to_autograd operator. (Pull request 16)

### Fixed
- Fixed to_autograd operator to always output 32 bit results, also with a 64 bit input just like the normal operator. Before the fix it only worked with 32 bit inputs.

## [0.5.0] - 2022-02-01
This release is mostly about maintaining existing functionality. The tomosipo paper and Astra toolbox 2 were released, there is a new maintainer, and some bugs were fixed. The conda package is now the default installation option.

### Added
- Added information and Bibtex entry of the tomosipo paper to the readme

### Changed
- Changed repository maintainer from Allard Hendriksen to Dirk Schut
- The conda package is now the default installation option, instead of cloning from Github with pip. This is changed in the readme
- Astra toolbox 2 was released so it is now the required version. There were no API changes compared to the recent development version that was required before.
- Removed cupy and odl packages from the environment presented in the documentation.
  Cupy was removed because of cudatoolkit error messages
  ODL was removed because it changes the global numpy formatting options

### Fixed
- Replaced usage of np.int which was deprecated in numpy 1.20 with int
- Replaced the usage of collect_ignore in setup.cfg because it was removed from pytest
- Licence information was updated to show GPLv3


## [0.4.1] - 2021-07-18
Bug fix release for `ts.rotate` (again)..

### Fixed
- `ts.rotate`: the rotation matrix had become transposed while fixing https://github.com/ahendriksen/tomosipo/issues/7. This has now been fixed.


## [0.4.0] - 2021-07-15
Bug fix release for `ts.rotate`. Minor bump for automatic torch and cupy
linking.

**NOTE**: `ts.rotate` is **broken** in this release. Please use v0.4.1.

### Changed
- It is no longer necessary to import `tomosipo.torch_support` or
  `tomosipo.cupy` to enable support for linking torch and cupy arrays. Support
  for these arrays is now added automatically when the packages are installed in
  the host environment.
### Fixed
- `ts.rotate` can now correctly deal with a changing axis of rotation over time.
  https://github.com/ahendriksen/tomosipo/issues/7
### Removed
- The `ts.forward` and `ts.backward` top-level bindings have been removed. They are
  still available as `ts.astra.forward` and `ts.astra.backward`.
- The `ts.fdk` top-level binding has been removed. It is still available as
  `ts.astra.fdk`.

## [0.3.1] - 2021-06-17
### Fixed
- `ts.operator`: making an operator with a volume vector geometry would fail.
  This has been fixed.

## [0.3.0] - 2021-06-17
In this release, mostly small things have been polished for consistency.
In addition, the transforms functions have gained some added
functionality to easily move and scale things over time.

### Added
- All geometry classes implement `len()`, which returns the number of steps in the geometry.
- Convenience methods `Transform.transform_vec` and `Transform.transform_point`
- `ts.svg()`: Render geometries to an animated SVG (no dependencies required(!)).
- `ts.svg()`: Added progress bar.
- `ts.svg()`: Optionally show z,y,x-axes in top-left corner with `show_axes=True`.
- `ts.reflect()`: Add reflection geometrical transform.
- `ts.from_perspective()` / `ts.to_perspective()`: Add `ignore_scale` parameter,
  which is `True` by default.
- `ts.scale()`: Add alpha parameter so that object can easily be scaled in
  certain directions over time.
- `ts.translate()`: Add alpha parameter so that object can easily be translated along a 
  certain axis over time.

### Changed
- `ts.svg()`: Remove `base64` keyword parameter. This functionality has been
  moved to the `_repr_markdown_` method.
- `ts.rotate()`: Disallow passing `pos` and `axis` by position. All
  parameters are now keyword-only.
- `ts.rotate()`: Deprecate `rad` and `deg` parameters in favor of
  `angles`. This makes `ts.rotate` more consistent with
  `ts.parallel()`, etc.
- `ts.from_perspective()` / `ts.to_perspective()`: All parameters are now keyword-only.
- `ts.from_perspective()` / `ts.to_perspective()`: The `box` parameter has been
  renamed to `vol`.
- `.to_box()`: Rename the `.to_box()` method on projection geometries to `.to_vol()`.


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



[Unreleased]: https://github.com/ahendriksen/tomosipo/compare/v0.5.1...develop
[0.5.1]: https://github.com/ahendriksen/tomosipo/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ahendriksen/tomosipo/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/ahendriksen/tomosipo/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ahendriksen/tomosipo/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/ahendriksen/tomosipo/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ahendriksen/tomosipo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ahendriksen/tomosipo/compare/v0.0.1...v0.2.0
[0.0.1]: https://github.com/ahendriksen/tomosipo/releases/tag/v0.0.1
