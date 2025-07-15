from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import correlation

from pyextal.LucyRichardson import DQE

class BaseGOF(ABC):
    name:str = NotImplemented
    @abstractmethod
    def __call__(self, simulation, experiment):
        pass


class XCorrelation(BaseGOF):
    name = "Cross Correlation"
    def __call__(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32])->np.float32:
        assert simulation.shape == experiment.shape
        return correlation(simulation.flatten(), experiment.flatten())


class Chi2(BaseGOF):
    '''
    Chi Square Goodness of Fit class with constant background for all disks and poisson noise only
    '''
    name = "Chi Square single background no detector"
        
    def __call__(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[np.bool_]=None)->np.float32:
        '''
        calculate the chi square goodness of fit with constant background for all disks
        '''
        
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
        self.sigma2 = np.abs(experiment)        
    
    def scaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[np.bool_]=None)->np.ndarray[np.float32]:
        '''
        scale the simulation to the experiment
        '''
        self.scale, self.background = self.calScaling(simulation, experiment, mask)
        
        scaled = simulation*self.scale
        scaled += self.background
        return scaled
    
    def calScaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[np.bool_]=None):
        '''
        find the scale (for all disks) and background (for each disk) that minimize the chisq

        $$\chi^2 = \sum_d \sum_i \frac{(cI_{id}^t+ b - I_{id}^x)^2}{\sigma_{id}^2}$$
        $$\frac{\partial \chi^2}{\partial c} = 2(c\sum_d \sum_i \frac{{I_{id}^t}^2}{\sigma^2_{id}} + b\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^t_{id}I^x_{id}}{\sigma^2_{id}})=0  $$
        $$ \frac{\partial \chi^2}{\partial b} = 2(c\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} + b\sum_d \sum_i \frac{1}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^x_{id}}{\sigma^2_{id}})=0$$

        solve for c and b     
        '''
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
    '''
    Chi Square Goodness of Fit
    '''
    name = "Chi Square background for each disk"
    def __init__(self, dinfo):        
        self.dinfo = dinfo

    def calVariance(self, experiment:np.ndarray[np.float32]):
        dqe = DQE(experiment, *self.dinfo.dtpar[:3])
        self.sigma2 = self.dinfo.dtpar[3]*experiment+self.dinfo.dtpar[4]*self.dinfo.dtpar[3]*(1/dqe - 1)
    
    def calScaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[np.bool_]=None):
        '''
        extal chisq.f subroutine tnorm0

        find the scale (for all disks) and background (for each disk) that minimize the chisq

        $$\chi^2 = \sum_d \sum_i \frac{(cI_{id}^t+ b_d - I_{id}^x)^2}{\sigma_{id}^2}$$
        $$\frac{\partial \chi^2}{\partial c} = 2(c\sum_d \sum_i \frac{{I_{id}^t}^2}{\sigma^2_{id}} + b\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^t_{id}I^x_{id}}{\sigma^2_{id}})=0  $$
        $$ \frac{\partial \chi^2}{\partial b_d} = 2(c\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} + b_d\sum_d \sum_i \frac{1}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^x_{id}}{\sigma^2_{id}})=0$$

        solve for c and then solve for b_d with substitution     
        '''
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
    '''
    Chi Square Goodness of Fit class with constant background for all disks
    '''
    name = "Chi Square single background"
    def __init__(self, dinfo):
        '''
        initialize the dinfo for the DQE and variance calculation
        '''
        self.dinfo = dinfo

    def calVariance(self, experiment:np.ndarray[np.float32]):
        dqe = DQE(experiment, *self.dinfo.dtpar[:3])
        self.sigma2 = self.dinfo.dtpar[3]*experiment+self.dinfo.dtpar[4]*self.dinfo.dtpar[3]*(1/dqe - 1)
    


class Chi2_LARBED(Chi2):
    '''
    Chi Square Goodness of Fit class with constant background for all disks with variance estimated for LARBED
    '''
    name = "Chi Square single bacckground LARBED"       
    
    def __init__(self, roi):
        '''
        initialize the roi for the variance
        '''
        self.roi = roi
    
    def calVariance(self, experiment:np.ndarray[np.float32]):
        '''
        override the variance calculation with the provided variance
        '''
        self.sigma2 = self.roi.variance


class Chi2_LARBED_multibackground(Chi2_LARBED):
    '''
    Chi Square Goodness of Fit class with differnt background for all disks with variance estimated for LARBED
    '''
    name = "Chi Square multiple bacckground LARBED"       
    
    def calScaling(self, simulation:np.ndarray[np.float32], experiment:np.ndarray[np.float32], mask:np.ndarray[np.bool_]=None):
        '''
        extal chisq.f subroutine tnorm0

        find the scale (for all disks) and background (for each disk) that minimize the chisq

        $$\chi^2 = \sum_d \sum_i \frac{(cI_{id}^t+ b_d - I_{id}^x)^2}{\sigma_{id}^2}$$
        $$\frac{\partial \chi^2}{\partial c} = 2(c\sum_d \sum_i \frac{{I_{id}^t}^2}{\sigma^2_{id}} + b\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^t_{id}I^x_{id}}{\sigma^2_{id}})=0  $$
        $$ \frac{\partial \chi^2}{\partial b_d} = 2(c\sum_d \sum_i \frac{I_{id}^t}{\sigma^2_{id}} + b_d\sum_d \sum_i \frac{1}{\sigma^2_{id}} - \sum_d\sum_i \frac{I^x_{id}}{\sigma^2_{id}})=0$$

        solve for c and then solve for b_d with substitution     
        '''
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
    
