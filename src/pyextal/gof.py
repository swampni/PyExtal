"""Goodness-of-Fit (GOF) Metrics Module.

This module provides a collection of classes for calculating goodness-of-fit
metrics between simulated and experimental diffraction data. It is designed to be
extensible, allowing for new GOF metrics to be added easily.

GOF Class Interface
-------------------
All GOF classes in this module are expected to follow a common interface to ensure
they can be used interchangeably throughout the refinement process. While not
formally enforced by an Abstract Base Class, this interface includes:

-   **`name` (str)**: A class attribute that provides a human-readable name for
    the metric (e.g., "Chi Square", "Cross Correlation").

-   **`__call__(self, simulation, experiment, mask=None)`**: The main method that
    calculates the GOF value. It takes the simulation and experiment as input
    and returns a single float value representing the goodness of fit.

-   **`scaling(self, simulation, experiment, mask=None)`**: An optional method
    that can be implemented by subclasses to scale the simulation intensity to
    the experimental intensity before the GOF calculation. If not implemented,
    no scaling is performed.

The `BaseGOF` class is provided as a simple parent class that new metrics can
inherit from, but this is not a requirement.
"""
import numpy as np
from scipy.spatial.distance import correlation

from pyextal.LucyRichardson import DQE
from pyextal.dinfo import BaseDiffractionInfo
from pyextal.roi import LARBEDROI

class BaseGOF:
    """Base class for Goodness-of-Fit (GOF) metrics.

    This class serves as a template and is not intended to be used directly.
    Subclasses should implement the `__call__` method.

    Attributes:
        name (str): The name of the GOF metric.
    """
    name:str = "Base GOF (Not Implemented)"

    def __call__(self, simulation: np.ndarray, experiment: np.ndarray, mask:np.ndarray[bool]=None):
        """Computes the goodness-of-fit between simulation and experiment.

        This method must be implemented by subclasses.

        Args:
            simulation (np.ndarray): The simulated data.
            experiment (np.ndarray): The experimental data.
            mask (np.ndarray[bool], optional): A boolean mask.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the `__call__` method.")


class XCorrelation(BaseGOF):
    """Calculates the cross-correlation between two datasets.

    This metric is a measure of similarity between two series as a function of the
    displacement of one relative to the other. It is computed using
    `scipy.spatial.distance.correlation`.
    """
    name = "Cross Correlation"
    def __call__(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32])->np.float32:
        """Calculates the correlation distance between simulation and experiment.

        Args:
            simulation (np.ndarray[np.float32]): The simulated diffraction pattern.
            experiment (np.ndarray[np.float32]): The experimental diffraction pattern.

        Returns:
            np.float32: The correlation distance.
        """
        assert simulation.shape == experiment.shape
        return correlation(simulation.flatten(), experiment.flatten())


class Chi2(BaseGOF):
    """Chi-squared goodness-of-fit with a single background and Poisson noise.

    This class calculates the chi-squared statistic assuming a constant background
    across all diffraction disks and that the noise follows a Poisson distribution.

    Attributes:
        name (str): The name of the GOF metric.
        sigma2 (np.ndarray): The variance of the experimental data.
    """
    name = "Chi Square single background no detector"
        
    def __call__(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[bool]=None)->np.float32:
        """Calculates the chi-squared value.

        Args:
            simulation (np.ndarray[np.float32]): The simulated diffraction pattern.
            experiment (np.ndarray[np.float32]): The experimental diffraction pattern.
            mask (np.ndarray[bool], optional): A boolean mask to include only
                specific regions in the calculation. Defaults to None.

        Returns:
            np.float32: The calculated chi-squared value.

        Raises:
            ValueError: If simulation and experiment arrays have different shapes.
        """
        
        if simulation.shape != experiment.shape:
            raise ValueError(f"simulation  and experiment should have the same shape but have {simulation.shape} and {experiment.shape}")
        scaled = self.scaling(simulation, experiment, mask)        
        if mask is not None:
            experiment = experiment[mask]
            scaled = scaled[mask]
            sigma2 = self.sigma2[mask]
        else:
            sigma2 = self.sigma2

        chisq = np.sum((scaled - experiment)**2/sigma2)
        return chisq

    def calVariance(self, experiment:np.ndarray[np.float32]):
        """Calculates the variance of the experiment, assuming Poisson noise.

        The variance is estimated as the absolute value of the experimental counts.

        Args:
            experiment (np.ndarray[np.float32]): The experimental data.
        """
        self.sigma2 = np.abs(experiment)        
    
    def scaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[bool]=None)->np.ndarray[np.float32]:
        """Scales the simulation to the experiment.

        Determines the optimal scale and background that minimizes chi-squared,
        then applies them to the simulation data.

        Args:
            simulation (np.ndarray[np.float32]): The simulated data.
            experiment (np.ndarray[np.float32]): The experimental data.
            mask (np.ndarray[bool], optional): A boolean mask to apply to the
                data. Defaults to None.

        Returns:
            np.ndarray[np.float32]: The scaled simulation data.
        """
        scale, background = self.calScaling(simulation, experiment, mask)
        
        scaled = simulation*scale
        scaled += background
        return scaled
    
    def calScaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[bool]=None):
        """Calculates the optimal scale and background to minimize chi-squared.

        Solves a system of linear equations to find the scale factor `c` and
        background `b` that minimize the chi-squared statistic:

        .. math::

            \chi^2 = \sum_d \sum_i \\frac{(cI_{id}^t+ b - I_{id}^x)^2}{\sigma_{id}^2}

        The derivatives with respect to `c` and `b` are set to zero:

        .. math::
            \\frac{\partial \chi^2}{\partial c} = 2(c\sum_d \sum_i \\frac{{I_{id}^t}^2}{\sigma^2_{id}} + b\sum_d \sum_i \\frac{I_{id}^t}{\sigma^2_{id}} - \sum_d\sum_i \\frac{I^t_{id}I^x_{id}}{\sigma^2_{id}})=0

        .. math::
            \\frac{\partial \chi^2}{\partial b} = 2(c\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} + b\sum_d \sum_i \frac{1}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^x_{id}}{\sigma^2_{id}})=0  

        Args:
            simulation (np.ndarray[np.float32]): The simulated data.  
            experiment (np.ndarray[np.float32]): The experimental data.  
            mask (np.ndarray[bool], optional): A boolean mask to apply to the
            data. Defaults to None.

        Returns:
            tuple[float, float]: The optimal scale and background values.
        """
        # ad hoc solution for interpolation error for very small values
        self.calVariance(experiment)
        
        if mask is not None:
            simulation = simulation[mask]
            experiment = experiment[mask]
            sigma2 = self.sigma2[mask]
        else:
            sigma2 = self.sigma2

        alpha = np.sum(simulation**2/sigma2)
        beta = np.sum(simulation/sigma2)
        gamma = np.sum(experiment*simulation/sigma2)
        
        epsilon = np.sum(experiment/sigma2)
        delta = np.sum(1/sigma2)

        scale = (gamma*delta - beta*epsilon)/(alpha*delta - beta**2)
        background = (alpha*epsilon - beta*gamma)/(alpha*delta - beta**2)
        return scale, background
class Chi2_multibackground(Chi2):
    """Chi-squared GOF with a separate background for each diffraction disk.

    Attributes:
        name (str): The name of the GOF metric.
        dinfo: a BaseDiffractionInfo object
    """
    name = "Chi Square background for each disk"
    def __init__(self, dinfo: BaseDiffractionInfo):
        """Initializes the Chi2_multibackground metric.

        Args:
            dinfo: An object containing diffraction information, including detector
            parameters (`dtpar`) needed for variance calculation.
        """
        self.dinfo = dinfo

    def calVariance(self, experiment:np.ndarray[np.float32]):
        """Calculates variance based on detector DQE.

        Args:
            experiment (np.ndarray[np.float32]): The experimental data.
        """
        dqe = DQE(experiment, *self.dinfo.dtpar[:3])
        self.sigma2 = self.dinfo.dtpar[3]*experiment+self.dinfo.dtpar[4]*self.dinfo.dtpar[3]*(1/dqe - 1)
    
    def calScaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[bool]=None):
        """Calculates scale and per-disk backgrounds to minimize chi-squared.

        This method is based on the `extal` `chisq.f` subroutine `tnorm0`.
        It finds a single scale factor `c` and a separate background `b_d` for
        each disk `d` that minimize the chi-squared statistic:

        .. math::
            \chi^2 = \sum_d \sum_i \\frac{(cI_{id}^t+ b_d - I_{id}^x)^2}{\sigma_{id}^2}

        The derivatives are set to zero and solved:

        .. math::
            \\frac{\partial \chi^2}{\partial c} = 2(c\sum_d \sum_i \\frac{{I_{id}^t}^2}{\sigma^2_{id}} + \sum_d b_d \sum_i \\frac{I_{id}^t}{\sigma^2_{id}} - \sum_d\sum_i \\frac{I^t_{id}I^x_{id}}{\sigma^2_{id}})=0

        .. math::
            \\frac{\partial \chi^2}{\partial b_d} = 2(c\sum_i \frac{I_{id}^t}{\sigma^2_{id}} + b_d\sum_i \frac{1}{\sigma^2_{id}} - \sum_i \frac{I^x_{id}}{\sigma^2_{id}})=0

        Args:
            simulation (np.ndarray[np.float32]): The simulated data.
            experiment (np.ndarray[np.float32]): The experimental data.
            mask (np.ndarray[bool], optional): A boolean mask to apply to the
            data. Defaults to None.

        Returns:
            tuple[float, np.ndarray]: The optimal scale factor and an array of
            background values for each disk.
        """
        self.calVariance(experiment)

        if mask is not None:
            simulation = simulation[mask]
            experiment = experiment[mask]
            sigma2 = self.sigma2[mask]
            sum1 = np.sum(simulation*experiment/sigma2)
            sum2 = np.sum(simulation**2/sigma2)
            cut = np.cumsum(np.sum(mask, axis=1)[:-1])
            simulation = np.split(simulation, cut)
            experiment = np.split(experiment, cut)
            sigma2 = np.split(sigma2, cut) 
        else:
            sigma2 = self.sigma2          
            sum1 = np.sum(simulation*experiment/sigma2)
            sum2 = np.sum(simulation**2/sigma2)       
               
        
        sk = []
        ak = []
        bk = []
        for i in range(len(simulation)):
            sk.append(np.sum(1/sigma2[i]))
            ak.append(np.sum(experiment[i]/sigma2[i]))
            bk.append(np.sum(simulation[i]/sigma2[i]))
            sum1 -= ak[i]*bk[i]/sk[i]
            sum2 -= bk[i]*bk[i]/sk[i]
        scale = sum1/sum2
        background = []
        for i in range(len(simulation)):
            background.append((ak[i] - scale*bk[i])/sk[i])
        background = np.array(background).reshape(len(simulation), 1)
        return scale, background
    
class Chi2_const(Chi2):
    """Chi-squared GOF with a single background and DQE-based variance.

    Attributes:
        name (str): The name of the GOF metric.
        dinfo: An object containing diffraction information, used for DQE calculation.
    """
    name = "Chi Square single background"
    def __init__(self, dinfo):
        """Initializes the Chi2_const metric.

        Args:
            dinfo: An object containing diffraction information, including detector
                parameters (`dtpar`) needed for variance calculation.
        """
        self.dinfo = dinfo

    def calVariance(self, experiment:np.ndarray[np.float32]):
        """Calculates variance based on detector DQE.

        Args:
            experiment (np.ndarray[np.float32]): The experimental data.
        """
        dqe = DQE(experiment, *self.dinfo.dtpar[:3])
        self.sigma2 = self.dinfo.dtpar[3]*experiment+self.dinfo.dtpar[4]*self.dinfo.dtpar[3]*(1/dqe - 1)
    


class Chi2_LARBED(Chi2):
    """Chi-squared GOF for LARBED data with a single background.

    This class uses a pre-calculated variance map, specific to LARBED experiments.

    Attributes:
        name (str): The name of the GOF metric.
        roi: A LARBEDROI object, including the variance map.
    """
    name = "Chi Square single background LARBED"

    def __init__(self, roi: LARBEDROI):
        """Initializes the Chi2_LARBED metric.

        Args:
            roi: An object containing the Region of Interest, which must have a
                `variance` attribute.
        """
        self.roi = roi
    
    def calVariance(self, experiment:np.ndarray[np.float32]):
        """Sets the variance from the pre-calculated LARBED variance map.

        Args:
            experiment (np.ndarray[np.float32]): The experimental data (not used,
                but maintained for compatibility).
        """
        self.sigma2 = self.roi.variance


class Chi2_LARBED_multibackground(Chi2_LARBED):
    """Chi-squared GOF for LARBED with per-disk backgrounds.

    Combines the pre-calculated variance from `Chi2_LARBED` with the per-disk
    background calculation from `Chi2_multibackground`.

    Attributes:
        name (str): The name of the GOF metric.
    """
    name = "Chi Square multiple bacckground LARBED"       
    
    def calScaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[bool]=None):
        """Calculates scale and per-disk backgrounds for LARBED data.

        This method is an alias for the multi-background scaling calculation,
        but it uses the pre-calculated variance from the LARBED ROI.

        Args:
            simulation (np.ndarray[np.float32]): The simulated data.
            experiment (np.ndarray[np.float32]): The experimental data.
            mask (np.ndarray[bool], optional): A boolean mask to apply to the
                data. Defaults to None.

        Returns:
            tuple[float, np.ndarray]: The optimal scale factor and an array of
            background values for each disk.
        """
        self.calVariance(experiment)

        if mask is not None:
            simulation = simulation[mask]
            experiment = experiment[mask]
            sigma2 = self.sigma2[mask]
            sum1 = np.sum(simulation*experiment/sigma2)
            sum2 = np.sum(simulation**2/sigma2)
            cut = np.cumsum(np.sum(mask, axis=1)[:-1])
            simulation = np.split(simulation, cut)
            experiment = np.split(experiment, cut)
            sigma2 = np.split(sigma2, cut) 
        else:
            sigma2 = self.sigma2          
            sum1 = np.sum(simulation*experiment/sigma2)
            sum2 = np.sum(simulation**2/sigma2)       
               
        
        sk = []
        ak = []
        bk = []
        for i in range(len(simulation)):
            sk.append(np.sum(1/sigma2[i]))
            ak.append(np.sum(experiment[i]/sigma2[i]))
            bk.append(np.sum(simulation[i]/sigma2[i]))
            sum1 -= ak[i]*bk[i]/sk[i]
            sum2 -= bk[i]*bk[i]/sk[i]
        scale = sum1/sum2
        background = []
        for i in range(len(simulation)):
            background.append((ak[i] - scale*bk[i])/sk[i])
        background = np.array(background).reshape(len(simulation), 1)
        return scale, background

