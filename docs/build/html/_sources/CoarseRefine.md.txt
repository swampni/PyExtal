# Coarse Refinement (Correlation based refinement)  
In order to reduce calculation time, some parameters can be optimized with correlation metric, especially for those parameters that are sensitive to feature positions (e.g. holz line, rocking curve spacing etc.). This is done by simulating intensity of a predefined range of tilt wider than experiment field of view in reciprocal space using given parameter. The simulation can be done with a grid coarser than experiment data to reduce computation time. Experiment data is then used as a template to find the tilt position with highest correlation coefficient. This identify the best incident beam direction. In pyExtal, we applied the `features.match_template` function in [scikit-image](https://scikit-image.org/) library for the matching. This process is defined as the target function, which is feed into `scipy.optimize.minimize` function with `method='brent'` option for optimization. All the optimization is done in the **display** space as described in [ROI]()

To perform the coarse refinement, you need to set up a `CoarseRefine` object with the following parameters:
```python
from pyextal.optimize import CoarseOptimize
coarse = CoarseOptimize(datpath='path/to/dat', dinfo=dinfo, roi=roi)
```


## image preprocessing  
 Sometimes it is also advisable to use some image filtering techniques to enhance the features of interest. We provided a argument that allows user to pass in any kind of image preprocessing function with following singature to enhance the features.  
```python
def preprocess(image: np.ndarray, sim=False) -> np.ndarray:
    """
    Preprocess the input image to enhance features for correlation matching. 
    Simulation and experiment patterns may require different preprocessing techniques as the range of intensity is different. 
    Normalizaltion might be needed for thresholding or filtering.
    
    Parameters:
    - image: Input image as a numpy array.
    - sim: Boolean indicating if the image is a simulation pattern (default is False).
    
    Returns:
    - Preprocessed image as a numpy array.
    """
    if sim:
        res = some_simulation_preprocessing_function(image)
    else:
        res = some_experiment_preprocessing_function(image)
    return res
```  
This provides great flexibility for users to apply their own image processing techniques. Some predefined preprocessing functions will be shipped with pyExtal in the future.  

## geometry refinement  
Geometry parmeters inlcude the incident beam direction, step size, and sample thicknss. In pyExtal, step size and sample thickness can only be refined separately and simultaneously with sample thickness. This is due to the fact that both step size and thickness affect the features in the diffraction pattern, so doing them together will lead to convergence issues. 
- `CoarseRefine.optimizeOrientationThickness`: This function optimizes the target function using thickness as input parameter.  
- `CoarseRefine.optimizeOrientationGL`: This function optimizes the target function using step size (gl) as input parameter. 

## HV calibration  
In CBED/LARBED experiments, the accceleration voltage (HV) is a critical parameter that affects the diffraction pattern. Similar to geometry parameter, a target function taking hV as input can be optimized. The `CoarseRefine.optimizeHV` function is used to optimize the HV parameter.  


## Debye-Waller factor  
Debye-Waller factor is a parameter that describes the thermal vibrations of atoms in the crystal lattice. It can be refined using the coarse refinement method. The `CoarseRefine.optimizeDWF` function is used to optimize the Debye-Waller factor. This function takes the target function and optimizes it with respect to the Debye-Waller factor.

## unit cell refinement  
Unit cell parameters can also be refined using the coarse refinement method. The `CoarseRefine.optimizeUnitCell` function is used to optimize the unit cell parameters. This function takes the target function and optimizes it with respect to the unit cell parameters. As we have 6 unit cell parameters, the optimization is done with respect to all 6 parameters simultaneously using `method='Nelder-Mead'` option in `scipy.optimize.minimize`.