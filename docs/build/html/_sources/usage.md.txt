# Getting started with PyExtal

Currently, the package is only available on linux. For windows users, we recommend using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) to run the package. All the development and testing is done on Ubuntu 22.04 with WSL.  

PyExtal used conda as package manager. Please make sure you have conda installed. You can install conda by following the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

## Installation

To install the package, you can use conda:
```bash
conda install hcni2::pyextal
```
This will install the package and all its dependencies. It is highly recommended to install the package in a virtual environment to avoid conflicts with other packages.


## FAQ  

1. ModuleNotFoundError: No module named 'distutils.msvccompiler'?  

    This could be a problem with python package setuptools. I fixed this with
    ```bash
    conda install "setuptools<65"
    ```
2. Issue related with finding MPI

    This is often related to the MPI library not being found. If you are using Intel compiler, you need to source the environment variables for the MPI library. You can do this by running the following command:
    ```bash
    source /opt/intel/oneapi/mpi/latest/env/vars.sh (or where you install your mpi library)
    ```  
3. matplotlib qt5 backend error
    
    ```bash
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: eglfs, minimal, minimalegl, offscreen, vnc, webgl, xcb.

    Aborted (core dumped)
    ```

    This is probably a WSL issue. You can either use another backend, e.g. tk, or try to install PyQt5 dependences with running the following command:    
    
    ```bash
    sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
    ```