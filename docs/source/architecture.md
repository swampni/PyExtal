# Package Architecture  

PyExtal is composed of two main components:  
1. [Bloch wave simulation engine](#bloch-wave-simulation-engine)
2. [Interface for optimization and analysis](#interface-for-optimization-and-analysis)

## Bloch Wave Simulation Engine  

Bloch Wave Simulation Engine is responsible for simulating diffraction intensity of a crystal structure. It is implemented in Fortran and it is adapted from the code described in *Electron Microdiffraction* by [Zuo and Spence](https://doi.org/10.1007/978-1-4899-2353-0). Diagonalization calculation is performed using [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.lup97p) implementation of the LAPACK library. Calculation of the diffraction intensity with different incident angle is parallelized using [Message Passing Interface (MPI)](https://www.mpi-forum.org/), enabling efficient computation. The Fortran code is compiled into a shared library using [f2py](https://numpy.org/doc/stable/f2py/) and is called from Python. The Fortran code is located in the `src/bloch` directory.  

The engine has three main subroutines:

- `bloch_parse`:  
Reads the input file and parses the parameters for crystal structure, experiment and simulation conditions. This must be called first before any other subroutines as it initializes the parameters.

- `bloch_run`/`tilt_run`:
Performs the Bloch wave simulation. Based on the input parameters, `bloch_run` will determine a set of incident directions and solve for eigenvalue and eigenvector for each direction. The `tilt_run` subroutine accepts a set of predefined incident directions and solve for eigenvalue and eigenvector for each direction, which is useful for refinement where we need to specficy the incident directions.  

**Parallelization Structure**:  
In order to have a seamless integration with Jupyter Notebook, we adopted a host-client structure for parallelization. All operations in python and the three subroutines mentioned here run on the main process (parent group with only one process). When `bloch_run`/`tilt_run` is called the first time, it will spawn new processes (children group) using `MPI_Comm_Spawn` running another subroutine `parabloch`. The `parabloch` subroutine will perform the actual calculation and return the results (eigenvector and eigenvalues of the A matrix) to the main process. After that, the children group will be parked and wait for input from the main process. This structure allows us to isolate the parallelization in the children processes, and users don't need to handle the paralleization. This allows users to work on Jupyter Notebook without the dealing with the complexity of working with parallelization.

- `LARBED`:
Performs the intensity calculation for a given set of eigenvectors and eigenvalues. This allows us to calculate the diffraction intensity of different thickness without redoing the expensive calculation of eigenvectors and eigenvalues. The calculation is described by equ 5.17 in *Advanced electron microscopy* by Zuo and Spence.  

- `cbedp`:
This is a legacy subroutine that has the same function as `LARBED`. It is still used for some part of the code, but it is subject to be removed in the future.  

## Interface for Optimization and Analysis  
The interface for optimization and analysis is implemented in Python in a object-oriented fashion. It is designed to be user-friendly and easy to use. The interface is composed of several modules, each responsible for a specific task. The main modules are:  

### `dinfo`:  
This is a set of classes that contains the information of the diffraction pattern, including the pattern itself, initial esitmates of sample thickness, sample orientation, calibration of the pattern pixel size and path to the input file containing crystal and simulation parameters. For LARBED pattern, it also accepts the pre-calculated pixel-wise variance of intensity. It also contains methods for managing structure factors, so users can access the refined structure factors.  

### `ROI`:  
This is a set of classes that contains the information of the region of interest (ROI) in the diffraction pattern. It is used to define the area of the diffraction pattern that will be used for optimization. This class also manages the connection between the simulation and the experimental data.  

### `optimize`:  
This module contains two classes: Coarse and Fine.  
- Coarse Refinement: The Coarse refinement refines the parameters by comparing the experimental data with the simulated data with cross-correlation. The simulation is done with a grid sparser than the experimental data and upsampled to reduce the computation time. A set of predefine optimzation traget functions are defined.  
    - thickness: thickness of the sample
    - gl: scaling factor for the diffraction pattern, equivalent to camera length for CBED and tilting step size for LARBED
    - DWF: Debye-Waller factor, which is used to account for the thermal vibration of atoms in the crystal structure. The DWF is defined as $exp(-Bsin^2(\theta)/\lambda^2)$, where B is the Debye-Waller factor, theta is the scattering angle and lambda is the wavelength of the incident beam. The DWF is used to correct the intensity of the diffraction spots for thermal motion due to inelastic scattering.
    - cell: unit cell parameters, including the lattice parameters and the angles between the axes. The unit cell parameters are used to define the crystal structure and are used to calculate the diffraction pattern.
    - orientation: orientation of the crystal structure with respect to the incident beam. Refining this doesn't require extra calculation, instead, a field of view larger than the roi is simulated and the orientation is refined by finding the position that gives best match with the experimental data. This is often in tandem with other parameters, such as thickness and gl.

- Fine Refinement: fine refinement simulated the intensity pixel by pixel. Several different goodness-of-fit (GOF) metric are defined in the [`gof`](#gof) module. Optimization is performed using the `scipy.optimize` module and the default optimizer is the Nelder-Mead simplex algorithm. The fine refinement is solely for structure factor refinement. For each iteration, the structure factors are adjusted and the simulation is performed. Subsequently, the scaling factor(gl), thickness and orientation are also refined to give the best fit based on the defined GOF metric. The fine refinement will stop once the stopping criteria is met. 

### `gof`:  
This module contains several classes that define different goodness-of-fit (GOF) metrics. The GOF metrics are used to evaluate the quality of the fit between the experimental data and the simulated data. Users can define their own GOF metrics by inheriting from the base class `BaseGOF`. `BaseGOF` is an abstract base class that defines the interface for the GOF metrics.   


## Basic Workflow  
1. Prepare dataset:  
    - CBED: Energy-filtered CBED patterns should be indexed and deconvolved to remove the MTF of detector. This package comes with a Lucy-Richardson deconvolution implementation that can be used to deconvolve the CBED patterns. The deconvolved patterns should be saved in a directory.
    - LARBED: LARBED patterns should be extracted from the raw data and saved in a directory. The variance of the intensity for each pixel should also be calculated and saved. Detail description of the variance calculation is described elsewhere.
2. define input .dat file: .dat file is a configuration file that contains the crystal structure, simulation parameters and diffraction gemometry.
3. User should first create a `dinfo` object with the input file. CBED/LARBED pattern also needs to be inputed.  
4. Then, a `ROI` object should be created to define the region of interest in the diffraction pattern.  
5. After that, a `CoarseOptimization` object can be created to perform the coarse  refinement.  
6. Subsequently, a `FineOptimization` object can be created to perform the fine refinement. The final results can be accessed from the `dinfo` object.  


## detail explaination of the workflow
```{toctree}
:maxdepth: 1
LARBEDindex
configuration
ROI
CoarseRefine
FineRefine
```