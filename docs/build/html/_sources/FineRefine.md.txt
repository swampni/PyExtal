# Fine Refientment (Intensity based Refinement)  
To accurately determine the strucuture factors, the fine refinement method is used. This method uses the intensity of the diffraction pattern to refine the structure factors. The fine refinement is done in the **display space** as described in [ROI](https://pyextal.readthedocs.io/en/latest/ROI.html).  
Similar to the coarse refinement, a target function is defined to be optimized. The target function calculates the goodness-of-fit metric between experiment and simulated intensity. specified by the user. The target function is then optimized using `scipy.optimize.minimize` function with `method='Nelder-Mead'` option.  
The input of the target function is the structure factors, which are adjusted in each iteration. Users need to set which reflections are refined and provide their initial value. If no initial value is provided, independent atom modle (IAM) value will be used as initial value. In addition, users also need to specify a range for each reflections that are used for normalization during refinement. During optimization, the structure factors are normalized to the range specified by the user. This is done to ensure that the structure factors are within a reasonable range and to avoid numerical instability.    

$$SF_{normalized} = \frac{(SF - SF_{min})}{SF_{max} - SF_{min}}  $$  


The target function is consisted of two parts: the first part is invoking the reevaluation of the eigenvectors and eigenvalues based on the structure factors, and the second part is an optimization loop for all the geometric parameters, including rotation, step size, thickness and orientation based on the goodness-of-fit metric. Orientation refinement is optional, and there is an option that allows individual disks to be move independently for CBED patterns. The optimization is done with `scipy.optimize.minimize` function with `method=Powell` option.  
For LARBED patterns, there is an additional option that allows the user to specify a 2d voigt function with $\gamma$ (for lorentizan) and $\sigma$ (for gaussian) to model the point spread function in reciprocal space. The voigt function is convolved with the simulated intensity before the goodness-of-fit metric is calculated. This is critical for obtaining accurate structure factors for LARBED patterns, as the point spread function can significantly affect the intensity distribution in reciprocal space. $\gamma$ and $\sigma$ are refined together with the geometric parameters. 


## GOF Metrics
The following GOF metrics are implemented:  
- `XCorrelation`: return correlation calculated using `scipy.spatial.distance.correlation`  
- `Chi2`: the base class for $\chi^2$ metric which simply assumes poisson noise, so the variance is equal to the intensity.  

$$\chi^2 = \frac{1}{n-p-1} \sum_i \frac{(cI^t_i + b - I^{exp}_i)^2}{\sigma^2}  = \sum_i \frac{(cI^t_i + b - I^{exp}_i)^2}{I^{exp}_i} $$  
where $I^t_i$ is the simulated intensity, $I^{exp}_i$ is the experimental intensity, $c$ is the scaling factor, $b$ is the background and $\sigma^2$ is the variance. b and c are obtained by minimizing the $\chi^2$ metric with setting the derivative of $\chi^2$ with respect to $c$ and $b$ to zero. The equations are: 

$$\frac{\partial \chi^2}{\partial c} = 2(c\sum_i \frac{{I_{i}^t}^2}{\sigma^2_{i}} + b \sum_i \frac{I_{i}^t}{\sigma^2_{i}} - \sum_i \frac{I^t_{i}I^{exp}_{i}}{\sigma^2_{i}})=0$$

$$ \frac{\partial \chi^2}{\partial b} = 2(c \sum_i \frac{I_{i}^t}{\sigma^2_{i}} + b \sum_i \frac{1}{\sigma^2_{i}} - \sum_i \frac{I^{exp}_{i}}{\sigma^2_{i}})=0$$

- `chi2_const`: this is a subclass of `Chi2` that assumes the background is constant across the diffraction pattern. The variance is calculated based on the intensity and characteristics parameters of the detector input by user. The detector parameters are described in [Zuo (2000)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0029(20000501)49:3%3C245::AID-JEMT4%3E3.0.CO;2-O) 


- `chi2_multibackground`: this is a subclass of `Chi2_const` that allows multiple background values. This is useful for cases where the background is not uniform across different diffraction disks. An example is CBED patterns collected with Gatan Image filter, where the anisochromatcity is not perfectly removed during alignment. $\chi^2$ is now defined as :

$$\chi^2 = \frac{1}{n-p-1} \sum_d \sum_i \frac{(cI_{id}^t+ b_d - I_{id}^x)^2}{\sigma_{id}^2}$$

the normalization factor c and a set of background values $b_d$ are obtained by solving follwoing set of equations:

$$\frac{\partial \chi^2}{\partial c} = 2(c\sum_d \sum_i \frac{{I_{id}^t}^2}{\sigma^2_{id}} + b_d\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^t_{id}I^x_{id}}{\sigma^2_{id}})=0  $$

$$ \frac{\partial \chi^2}{\partial b_d} = 2(c\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} + b_d\sum_d \sum_i \frac{1}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^x_{id}}{\sigma^2_{id}})=0$$


- `chi2_LARBED`: the variance calculation is more involved for LARBED patterns, so instead of inputing a set of detector parameters, the user can precalculate the variance for each pixel and input it as a numpy array to the dinfo class. The variance will be passed to the roi class and accessed by the GOF metric. The normalization factor c and background b are obtained by solving the same set of equations described in `chi2_const`.

- `chi2_LARBED_multibackground`: the variance calculation is more involved for LARBED patterns, so instead of inputing a set of detector parameters, the user can precalculate the variance for each pixel and input it as a numpy array to the dinfo class. The variance will be passed to the roi class and accessed by the GOF metric. The normalization factor c and background b are obtained by solving the same set of equations described in `chi2_multibackground`.  

- custom GOF metric: user can define their own GOF metric by inheriting from the `BaseGOF` class. The user-defined GOF metric must implement the `__call__` method, which takes the simulated intensity and experimental intensity as input and returns the GOF value.