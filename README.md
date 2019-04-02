<h1 align="center">CosmiQ Works Geospatial Data Processing Tools for ML</h1>
<p align="center">
<a href="http://www.cosmiqworks.org"><img src="http://www.cosmiqworks.org/wp-content/uploads/2016/02/cropped-CosmiQ-Works-Logo_R_RGB.png" width="350" alt="CosmiQ Works"></a>
<br>
<br>
<br>
<!-- <img align="center" src="https://img.shields.io/pypi/v/cw-eval.svg" alt="PyPI"> -->
<!-- <img align="center" src="https://img.shields.io/conda/vn/conda-forge/cw-eval.svg" alt="conda-forge"> -->
<img align="center" src="https://travis-ci.com/CosmiQ/cw-geodata.svg?branch=master" alt="build">
<img align="center" src="https://readthedocs.org/projects/cw-geodata/badge/" alt="docs">
<img align="center" src="https://img.shields.io/github/license/cosmiq/cw-geodata.svg" alt="license">
<!-- <img align="center" src="https://img.shields.io/docker/build/cosmiqworks/cw-eval.svg" alt="docker"> -->
<a href="https://codecov.io/gh/CosmiQ/cw-geodata"><img align="center" src="https://codecov.io/gh/CosmiQ/cw-geodata/branch/master/graph/badge.svg" /></a>
</p>

__This package is currently under active development. Check back soon for a mature version.__

- [Installation Instructions](#installation-instructions)
- [API Documentation](https://cw-eval.readthedocs.io/)
- [Dependencies](#dependencies)
- [License](#license)
---
This package is built to:
- Enable management and interconversion of geospatial data files without requiring understanding of coordinate reference systems, geospatial transforms, etc.
- Enable creation of training targets for segmentation and object detection from geospatial vector data (_i.e._ geojsons of labels) without requiring understanding of ML training target formats.

## Installation Instructions
Several packages require binaries to be installed before pip installing the other packages. We recommend creating a conda environment and installing dependencies there from [environment.yml](./environment.yml), then using `pip` to install this package.

First, clone this repo to your computer and navigate into the folder:
```
git clone https://github.com/cosmiq/cw-geodata.git
cd cw-geodata
```
Next, create a [conda](https://anaconda.com/distribution/) environment with dependencies installed as defined in the environment.yml file.
```
conda create env -n cw-geodata -f environment.yml
conda activate cw-geodata
```
Finally, use `pip` to install this package.
```
pip install .
```
For bleeding-edge versions (use at your own risk), `pip install` from the dev branch of this repository:
```
pip install --upgrade git+https://github.com/CosmiQ/cw-geodata.git@dev
```

## API Documentation
API documentation can be found [here](https://cw-geodata.readthedocs.io)

## Dependencies
All dependencies can be found in [environment.yml](./environment.yml)

## License
See [LICENSE](./LICENSE.txt).
