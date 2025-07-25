# PyExtal: Python Package for Quantitative Electron Crystallography

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://swampni.github.io/bloch-python/)

**PyExtal** is a Python package designed for the quantitative simulation and analysis of electron diffraction data, particularly for convergent beam electron diffraction (CBED) and large angle rocking beam electron diffraction (LARBED) patterns.

---

## 📖 Documentation

Full documentation is available at:  
👉 [https://swampni.github.io/bloch-python/](https://swampni.github.io/bloch-python/)

---

## 🚀 Installation

### Using Conda Package

Currently, PyExtal only supports Python 3.11 on Linux. It is tested on WSL2 with Ubuntu 22.02.  

Install using conda (installing in a virtual environment is highly recommended):
```bash
conda install hcni2::pyextal
```

### installing miniconda
If you don't have conda installed, you can install Miniconda by following the instructions at [Miniconda Installation](https://www.anaconda.com/download/success).  

To create a new conda environment and install PyExtal, you can use the following commands:

```bash
conda create -n pyextal python=3.11
conda activate pyextal
conda install hcni2::pyextal
```

## Example

### CBED
Check out the [example notebook](https://github.com/swampni/PyExtal/blob/publish/example/CBED/refinemnt_si04.ipynb) 

### LARBED
Check out the [example notebook](https://github.com/swampni/PyExtal/blob/publish/example/LARBED/refinemntLARBEDSi111sys0420.ipynb)



## License

This repository contains a Python package designed to interact with a high-performance computational engine. It is important to understand the distinct licensing terms that apply to different parts of this project:

Python Package (Source Code): All Python source code within this repository is licensed under the Mozilla Public License, Version 2.0 (MPL 2.0).

Bloch Engine (Proprietary Binary): The core computational engine, provided as a pre-compiled shared library (e.g., .so file), is proprietary software by EMLab Solutions, Inc. It is not open-sourced under the MPL 2.0 or any other open-source license. Its use is governed by a separate, proprietary license agreement, which will be provided alongside the binary
