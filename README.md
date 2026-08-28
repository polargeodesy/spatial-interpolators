# spatial-interpolators

Python tools for spatially interpolating data over Cartesian and spherical grids

## About

<table>
  <tr>
    <td><b>Version:</b></td>
    <td>
        <a href="https://pypi.python.org/pypi/spatial-interpolators/" alt="PyPI"><img src="https://img.shields.io/pypi/v/spatial-interpolators.svg"></a>
        <a href="https://github.com/polargeodesy/spatial-interpolators/releases/latest" alt="commits-since"><img src="https://img.shields.io/github/commits-since/polargeodesy/spatial-interpolators/latest"></a>
    </td>
  </tr>
  <tr>
    <td><b>Citation:</b></td>
    <td>
        <a href="https://doi.org/10.5281/zenodo.6773704" alt="zenodo"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6773704.svg"></a>
    </td>
  </tr>
  <tr>
    <td><b>Tests:</b></td>
    <td>
        <a href="https://spatial-interpolators.readthedocs.io/en/latest/?badge=latest" alt="Documentation Status"><img src="https://readthedocs.org/projects/spatial-interpolators/badge/?version=latest"></a>
        <a href="https://github.com/polargeodesy/spatial-interpolators/actions/workflows/python-request.yml" alt="Build"><img src="https://github.com/polargeodesy/spatial-interpolators/actions/workflows/python-request.yml/badge.svg"></a>
        <a href="https://github.com/polargeodesy/spatial-interpolators/actions/workflows/ruff-format.yml" alt="Ruff"><img src="https://github.com/polargeodesy/spatial-interpolators/actions/workflows/ruff-format.yml/badge.svg"></a>
    </td>
  </tr>
  <tr>
    <td><b>License:</b></td>
    <td>
        <a href="https://github.com/polargeodesy/spatial-interpolators/blob/main/LICENSE" alt="License"><img src="https://img.shields.io/github/license/polargeodesy/spatial-interpolators"></a>
    </td>
  </tr>
</table>

For more information: see the documentation at [spatial-interpolators.readthedocs.io](https://spatial-interpolators.readthedocs.io/)

## Installation

From PyPI:

```bash
python3 -m pip install spatial-interpolators
```

To include all optional dependencies:

```bash
python3 -m pip install spatial-interpolators[all]
```

Using `conda` or `mamba` from conda-forge:

```bash
conda install -c conda-forge spatial-interpolators
```

```bash
mamba install -c conda-forge spatial-interpolators
```

Development version from GitHub:

```bash
python3 -m pip install git+https://github.com/polargeodesy/spatial-interpolators.git
```

### Running with Pixi

Alternatively, you can use [Pixi](https://pixi.sh/) for a streamlined workspace environment:

1. Install Pixi following the [installation instructions](https://pixi.sh/latest/#installation)
2. Clone the project repository:

```bash
git clone https://github.com/polargeodesy/spatial-interpolators.git
```

3. Move into the `spatial-interpolators` directory

```bash
cd spatial-interpolators
```

4. Install dependencies and start JupyterLab:

```bash
pixi run start
```

This will automatically create the environment, install all dependencies, and launch JupyterLab in the [notebooks](./doc/source/notebooks/) directory.

## Dependencies

- [cython: C-extensions for Python](https://cython.org)
- [numpy: Scientific Computing Tools For Python](https://www.numpy.org)
- [scipy: Scientific Tools for Python](https://www.scipy.org/)

## Download

The program homepage is:  
<https://github.com/polargeodesy/spatial-interpolators>

A zip archive of the latest version is available directly at:  
<https://github.com/polargeodesy/spatial-interpolators/archive/main.zip>

## Disclaimer

This package includes software developed at NASA Goddard Space Flight Center (GSFC) and the University of Washington Applied Physics Laboratory (UW-APL).
It is not sponsored or maintained by the Universities Space Research Association (USRA), or NASA.
The software is provided here for your convenience but *with no guarantees whatsoever*.

## Contributing

This project contains work and contributions from the [scientific community](./CONTRIBUTORS.md).
If you would like to contribute to the project, please have a look at the [contribution guidelines](./doc/source/getting_started/Contributing.rst), [open issues](https://github.com/polargeodesy/spatial-interpolators/issues) and [discussions board](https://github.com/polargeodesy/spatial-interpolators/discussions).

## License

The content of this project is licensed under the [Creative Commons Attribution 4.0 Attribution license](https://creativecommons.org/licenses/by/4.0/) and the source code is licensed under the [MIT license](LICENSE).
