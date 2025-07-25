from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.ndimage import zoom
from scipy.spatial.distance import correlation
from scipy.signal import convolve2d
from scipy.special import wofz
from skimage.feature import match_template
import matplotlib.pyplot as plt

from pyextal.dinfo import BaseDiffractionInfo
from pyextal.roi import BaseROI, split_array_by_lengths, ROITYPE
from pyextal.gof import BaseGOF
from pyextal.callBloch import bloch_run, LARBED_tilt, LARBED, calibrateLARBED, lookupSF, updateUgh, updateSymUgh, tilt_run, tilt
from pyextal.Constants import constant

import pyextal.blochwave as blochwave


class CoarseOptimize:
    """Class for coarse optimization of thickness, orientation, and gl.

    Attributes:
        dinfo (BaseDiffractionInfo): Diffraction information object.
        param (SimParams): Bloch simulation from dinfo.
        roi (BaseROI): The region of interest object.
        thickness (float): The sample thickness from dinfo.
        gl (float): geometric scaling factor from dinfo.
        indices (List[int]): List of indices matching reflections to the exitwave.
        exitwave (np.ndarray): The simulated exitwave.
        side (float): The side length of the simulation used for tilt mapping.
        scale_factor (float): The scale factor from LARBED calibration between simulation and experiment.
        templates (np.ndarray): The experimental templates extracted from the ROI.
    """

    def __init__(self, dinfo: BaseDiffractionInfo, roi: BaseROI, searchRadius: float = None, nx: int = None):
        """Initializes the CoarseOptimize object.

        Args:
            dinfo (BaseDiffractionInfo): Diffraction information object.
            roi (BaseROI): refineROI class object.
            searchRadius (float, optional): Search radius for the optimization. 
                Defaults to None.
            nx (int, optional): Number of pixels in radius. Defaults to None.
        """

        self.dinfo = dinfo
        self.param = dinfo.lastParam
        self.roi = roi
        # check if only one region is selected
        if self.roi.regions.shape[0] != 1:
            raise ValueError('CoarseOptimize only works for single region')
        # check if the region is a rectangle
        if self.roi.regions[0, 0, 0] != self.roi.regions[0, 1, 0] or self.roi.regions[0, 0, 1] != self.roi.regions[0, 2, 1]:
            raise ValueError(
                'CoarseOptimize only works for rectangular region')

        # make sure that no super or undersampling is applied
        # originally need this for adjust gl, but now we don't need it?
        self.roi.regions[0, 3, 0] = self.roi.regions[0,
                                                     1, 1] - self.roi.regions[0, 0, 1]
        self.roi.regions[0, 3, 1] = self.roi.regions[0,
                                                     2, 0] - self.roi.regions[0, 0, 0]
        self.roi.selectROI(self.roi.regions)

        # display the experimental pattern
        fig, axes = plt.subplots(
            1, len(self.roi.templates), figsize=(len(self.roi.templates)*4, 4))
        if len(self.roi.templates) == 1:
            axes = [axes]
        for i, template in enumerate(self.roi.templates):
            axes[i].imshow(template.reshape(
                self.roi.regions[0, 3, 1], self.roi.regions[0, 3, 0]))
            axes[i].axis('off')
            axes[i].set_title(f"reflection {i+1}")

        # local copy of the thickness, update this thickness to the dinfo.thickness after optimization
        self.thickness = self.dinfo.thickness
        # local copy of the gl
        self.gl = self.dinfo.gl
        
        
        # initialize the bloch simulation parameter (param))
        # self.param = bloch_parse(datpath, self.dinfo.tiltX, self.dinfo.tiltY)
        
        # ratio, allSymBeams = self.dinfo.getAllSF()
        # updateSymUgh(allSymBeams, ratio, self.dinfo.symmetry)
        # updateUgh(reflections=[(1,-1,-1),(2,-2,-2)],
        #           values = np.array([[0.0473921, 0.889252E-03],[-0.893626E-03, 0],]),
        #           beamDict=self.dinfo.beamDict)
        if searchRadius:
            self.param.disks = searchRadius
        
        if nx:
            self.param.nx = nx
        self.param = bloch_run(self.param, ncores=constant['NCORES'])      

        # match index of the reflections to the exitwave
        self.indices = []
        # if self.roi.roitype == ROITYPE.LARBED:
        #     for n in range(len(self.roi.nthgx)):
        #         self.indices.append(self.roi.indices[n][0])
        #         roi.indices[n] = self.roi.indices[n][0]

        # else:
        for index in self.roi.gInclude:
            for i, hkl in enumerate(self.param.hklout.T):
                if hkl[0] == index[0] and hkl[1] == index[1] and hkl[2] == index[2]:
                    self.indices.append(i)
                    break

        self.exitwave = LARBED(self.param, self.dinfo.thickness)
        self.side, self.scale_factor = calibrateLARBED(self.param, self.gl)
        images = []
        for i in self.indices:
            images.append(
                zoom(self.exitwave[i, :, :], self.scale_factor).astype(np.float32))
        self.exitwave = np.array(images)
        self.templates = self.roi.templates.reshape(
            -1, self.roi.regions[0, 3, 1], self.roi.regions[0, 3, 0])

    

    def optimizeOrientationGeometry(self, target: Callable, targetValue: float) -> float:
        """Optimizes a single parameter (thickness or gl) and the sample orientation.

        This method uses `scipy.optimize.minimize_scalar` to find the optimal
        value for a given target function (e.g., thickness or gl). After finding
        the best parameter value, it determines the optimal sample tilt by finding
        the location of the maximum correlation in the template matching result.

        Args:
            target (Callable): The target function to minimize (e.g., `thicknessTarget`
                or `glTarget`).
            targetValue (float): The initial guess for the parameter being optimized.

        Returns:
            float: The optimized value for the target parameter.
        """
        self.res = minimize_scalar(target,
                                   args=(self,),
                                   method=constant['coarse_geometry']['method'],
                                   bracket=[targetValue+constant['coarse_geometry']['bracket'][0], targetValue+constant['coarse_geometry']['bracket'][1]],
                                   options=constant['coarse_geometry']['options'])
        # self.dinfo.thickness = self.res.x
        # self.thickness = self.res.x

        target(self.res.x, self)

        total = match_template(self.exitwave, self.templates)
        searchLength = self.exitwave[0, :, :].shape[0]
        self.loc = np.unravel_index(np.argmax(total), total.shape[1:])
        # construct tiltmap
        x = np.linspace(-self.side/2, self.side/2, searchLength+1)
        X, Y = np.meshgrid(x, x)

        # calculate the relative position of the dp center to the corner
        if self.roi.roitype == ROITYPE.CBED:
            center = -self.roi.regions[0, 0].astype(np.int16)
        else:
            center = self.roi.dpCenter - self.roi.regions[0, 0].astype(np.int16)
        # infer tilt of the dp center
        
        self.dinfo.tiltX += -X[self.loc[0] +
                                center[0], self.loc[1] + center[1]]
        self.dinfo.tiltY += Y[self.loc[0] + center[0], self.loc[1] + center[1]]
        
        # update the tilt
        tilt(self.dinfo.lastParam, -X[self.loc[0] +
                                center[0], self.loc[1] + center[1]], 
                                Y[self.loc[0] + center[0], self.loc[1] + center[1]])
        self.roi.updateSimGrid()

        # self.loc = np.array(self.loc) + center

        


        return self.res.x
    
    def optimizeOrientationThickness(self, filter=None, threshold=None) -> None:
        """Optimizes the thickness and orientation of the experimental pattern."""
        self.filter = filter
        self.threshold = threshold
        thickness = self.optimizeOrientationGeometry(self.thicknessTarget, self.thickness)        
        self.dinfo.thickness = thickness
        self.thickness = thickness        
        print(f"thickness: {self.dinfo.thickness}, gl: {self.dinfo.gl}, tiltY: {self.dinfo.tiltY}, tiltX: {self.dinfo.tiltX}", flush=True)

    def optimizeOrientationGL(self, filter=None, threshold=None) -> None:
        """Optimizes the gl and orientation of the experimental pattern."""
        self.filter = filter
        self.threshold = threshold
        gl = self.optimizeOrientationGeometry(self.glTarget, self.gl)
        self.dinfo.gl = gl
        self.gl = gl
        self.roi.gl = gl
        print(f"thickness: {self.dinfo.thickness}, gl: {self.dinfo.gl}, tiltY: {self.dinfo.tiltY}, tiltX: {self.dinfo.tiltX}", flush=True)
    
    @staticmethod
    def thicknessTarget(x0, *args):
        """Error function for thickness optimization.

        Args:
            x0 (float): Thickness value.
            *args: CoarseOptimize instance.

        Returns:
            float: The optimization error, calculated as 1 - max_correlation.
        """
        thickness = x0
        self = args[0]
        
        self.exitwave = LARBED(self.param, thickness)
        images = []
        for i in self.indices:
            images.append(
                zoom(self.exitwave[i, :, :], self.scale_factor).astype(np.float32))
        self.exitwave = np.array(images)
        
        exitwave = self.exitwave
        templates = self.templates
            
        if self.filter:
            exitwave = np.array([self.filter(image, sim=True) for image in exitwave])
            templates = np.array([self.filter(template)
                                  for template in templates])
        
        
        if self.threshold:
            exitwave = np.array([self.threshold(image) for image in exitwave])
            templates = np.array([self.threshold(template)
                                  for template in templates])
        
        
        total = match_template(exitwave, templates)
        return 1 - np.max(total)
    @staticmethod
    def glTarget(x0, *args):
        """Error function for gl optimization.

        Args:
            x0 (float): gl value.
            *args: CoarseOptimize instance.

        Returns:
            float: The optimization error, calculated as 1 - max_correlation.
        """
        self = args[0]
        self.exitwave = LARBED(self.param, self.thickness)
        _, self.scale_factor = calibrateLARBED(self.param, x0)
        images = []
        for i in self.indices:
            
            images.append(
                zoom(self.exitwave[i, :, :], self.scale_factor).astype(np.float32))
            
        self.exitwave = np.array(images)

        exitwave = self.exitwave
        templates = self.templates

        if self.filter:
            exitwave = np.array([self.filter(image, sim=True) for image in exitwave])
            templates = np.array([self.filter(template)
                                  for template in templates])       
        
        if self.threshold:
            exitwave = np.array([self.threshold(image) for image in exitwave])
            templates = np.array([self.threshold(template)
                                  for template in templates])

        total = match_template(exitwave, templates)
        return 1 - np.max(total)
    


        
    

    def optimizeDWF(self) -> None:
        """Optimizes the Debye-Waller factors (DWFs) for all atoms.

        This method uses `scipy.optimize.minimize` to find the optimal DWF values
        that maximize the correlation between simulated and experimental patterns.
        It assumes isotropic DWFs.
        """
        y0 = blochwave.cryst.dw[:blochwave.cryst.natoms]
        boundary = []
        for i in range(blochwave.cryst.natoms):
            boundary.append(constant['coarse_DWF']['boundary'])
        self.res = minimize(self.correlationTargetDWF, y0, args=(self,),
                        method=constant['coarse_DWF']['method'],
                        bounds=boundary,
                        callback=self.callbackDWF,
                        options=constant['coarse_DWF']['options'])   
        self.dw = y0
        blochwave.cryst.dw[:blochwave.cryst.natoms] = y0
        print(
            f"DWF: {self.dw}")
    @staticmethod
    def callbackDWF(y0):
        print(y0)

    @staticmethod
    def correlationTargetDWF(y0, *args):
        """Error function for DWF optimization.

        Args:
            y0 (np.ndarray): Debye-Waller factors.
            *args: a CoarseOptimize instance.

        Returns:
            float: The optimization error, calculated as 1 - max_correlation.
        """
        
        self = args[0]
        #Isotropic only for now.
        for i in range(len(y0)):
            # for aborption part
            blochwave.cryst.dw[i] = y0[i]
            # for elastic part
            blochwave.cryst.atpar[3][i] = (blochwave.cryst.dw[i]/4/self.param.gmx[0][0])
            blochwave.cryst.atpar[4][i] = (blochwave.cryst.dw[i]/4/self.param.gmx[1][1])
            blochwave.cryst.atpar[5][i] = (blochwave.cryst.dw[i]/4/self.param.gmx[2][2])

        blochwave.resetugh()
        #self.param = bloch_run(self.param, ncores=constant['NCORES'])
        self.param = bloch_run(self.param, ncores=constant['NCORES'])  
        self.dinfo.tiltX, self.dinfo.tiltY = 0.0, 0.0
        self.optimizeOrientationThickness()   
        self.exitwave = LARBED(self.param, self.thickness)
        images = []
        for i in self.indices:
            images.append(
                zoom(self.exitwave[i, :, :], self.scale_factor).astype(np.float32))
        self.exitwave = np.array(images)
        total = match_template(self.exitwave, self.templates)
        return 1 - np.max(total)
    
    def optimizeXYZ(self) -> None:
        """Optimizes the atomic (x, y, z) coordinates for all atoms.

        This method uses `scipy.optimize.minimize` to find the optimal atomic
        positions that maximize the correlation between simulated and
        experimental patterns.
        """
        y0 = []
        for i in range(blochwave.cryst.natoms):
            y0.append(blochwave.cryst.atpar[0][i])
            y0.append(blochwave.cryst.atpar[1][i])
            y0.append(blochwave.cryst.atpar[2][i])
        boundary = []
        for i in range(len(y0)):
            boundary.append((y0[i]+constant['coarse_XYZ']['boundary'][0],y0[i]+constant['coarse_XYZ']['boundary'][1]))
        self.res = minimize(self.correlationTargetXYZ, y0, args=(self,),
                        method=constant['coarse_XYZ']['method'],
                        bounds=boundary,
                        #callback=lambda intermediate_result: self.callback(
                        #intermediate_result, self),
                        options=constant['coarse_XYZ']['options'])   
        '''for i in range(blochwave.cryst.natoms):
            blochwave.cryst.atpar[0][i] = y0[i*3]
            blochwave.cryst.atpar[1][i] = y0[i*3+1]
            blochwave.cryst.atpar[2][i] = y0[i*3+2]'''
        print(
            f"DWF: {self.dw}")
    @staticmethod
    def callbackXYZ(y0):
        print(y0)

    @staticmethod
    def correlationTargetXYZ(y0, *args):
        """Error function for XYZ coordinate optimization.

        Args:
            y0 (np.ndarray): XYZ coordinates for each atom.
            *args: a CoarseOptimize instance.

        Returns:
            float: The optimization error, calculated as 1 - max_correlation.
        """
        
        self = args[0]
        for i in range(blochwave.cryst.natoms):
            blochwave.cryst.atpar[0][i] = y0[i*3]
            blochwave.cryst.atpar[1][i] = y0[i*3+1]
            blochwave.cryst.atpar[2][i] = y0[i*3+2]

        blochwave.resetugh()
        #self.param = bloch_run(self.param, ncores=constant['NCORES'])
        self.param = bloch_run(self.param, ncores=constant['NCORES'])  
        self.dinfo.tiltX, self.dinfo.tiltY = 0.0, 0.0
        self.optimizeOrientationThickness()   
        self.exitwave = LARBED(self.param, self.thickness)
        images = []
        for i in self.indices:
            images.append(
                zoom(self.exitwave[i, :, :], self.scale_factor).astype(np.float32))
        self.exitwave = np.array(images)
        total = match_template(self.exitwave, self.templates)
        return 1 - np.max(total)

    def optimizeCell(self) -> None:
        """Optimizes the cell parameters.

        Currently, this method optimizes 'a' and 'c' for a tetragonal/hexagonal
        system, assuming a=b. It uses `scipy.optimize.minimize` to find the
        optimal cell parameters that maximize the correlation.
        """
        y0 = []
        y0.append(blochwave.cryst.cell[0])
        y0.append(blochwave.cryst.cell[2])
        boundary = []
        for i in range(len(y0)):
            boundary.append((y0[i]-0.15,y0[i]+0.15))
        self.res = minimize(self.correlationTargetCell, y0, args=(self,),
                        method='Nelder-Mead',
                        bounds=boundary,
                        #callback=lambda intermediate_result: self.callback(
                        #intermediate_result, self),
                        options={'maxiter': 10000, 'disp': True, 'fatol': 1e-5, 
                                 'xatol':1e-5, 'adaptive':True})   
        '''for i in range(blochwave.cryst.natoms):
            blochwave.cryst.atpar[0][i] = y0[i*3]
            blochwave.cryst.atpar[1][i] = y0[i*3+1]
            blochwave.cryst.atpar[2][i] = y0[i*3+2]'''
        print(
            f"Cell: {self.res.x}")
    @staticmethod
    def callbackCell(y0):
        print(y0)

    @staticmethod
    def correlationTargetCell(y0, *args):
        """Error function for cell parameter optimization.

        Args:
            y0 (np.ndarray): Cell parameters.
            *args: a CoarseOptimize instance.

        Returns:
            float: The optimization error, calculated as 1 - max_correlation.
        """
        
        self = args[0]
        blochwave.cryst.cell[0] = y0[0]
        blochwave.cryst.cell[1] = y0[0]
        blochwave.cryst.cell[2] = y0[1]

        blochwave.resetugh()
        print(blochwave.cryst.cell)
        #self.param = bloch_run(self.param, ncores=constant['NCORES'])
        self.param = bloch_run(self.param, ncores=constant['NCORES'])  
        self.dinfo.tiltX, self.dinfo.tiltY = 0.0, 0.0
        self.optimizeOrientationThickness()   
        self.exitwave = LARBED(self.param, self.thickness)
        images = []
        for i in self.indices:
            images.append(
                zoom(self.exitwave[i, :, :], self.scale_factor).astype(np.float32))
        self.exitwave = np.array(images)
        total = match_template(self.exitwave, self.templates)
        return 1 - np.max(total)
    
    def optimizeHV(self, filter=None, threshold=None) -> None:
        """Optimizes the high voltage (HV)."""
        self.filter = filter
        self.threshold = threshold
        # self.hvHistory = []
        # self.corrHistory = []
        self.res = minimize_scalar(self.HVTarget,
                                    args=(self,),
                                    method='brent',
                                    bracket=[blochwave.difpar.kv -
                                             0.2, blochwave.difpar.kv+0.2],
                                    options={'maxiter': 200, 'disp': True, 'xtol': 1e-5})
        print(f"final HV: {self.res.x}")
        blochwave.difpar.updatekv(self.res.x)
        blochwave.resetugh()
        self.param = bloch_run(self.param, ncores=constant['NCORES'])
        self.optimizeOrientationGL(filter=filter, threshold=threshold)
    
    @staticmethod
    def HVTarget(x0, *args):
        """Error function for HV optimization.

        Args:
            x0 (float): High voltage value.
            *args: a CoarseOptimize instance.

        Returns:
            float: The correlation error.
        """
        self = args[0]
        print(f'HV: {x0}kV',)
        # self.hvHistory.append(x0)
        blochwave.difpar.updatekv(x0)
        blochwave.resetugh()
        self.param = bloch_run(self.param, ncores=constant['NCORES'])
        # res = self.glTarget(self.gl, self)
        # self.optimizeOrientationThickness(filter=self.filter, threshold=self.threshold)
        self.exitwave = LARBED(self.param, self.thickness)
        images = []
        for i in self.indices:
            images.append(
                zoom(self.exitwave[i, :, :], self.scale_factor).astype(np.float32))
        self.exitwave = np.array(images)

        # TODO: this pattern should be a function
        exitwave = self.exitwave
        templates = self.templates
        h,w = templates[0].shape

        if self.filter:
            exitwave = np.array([self.filter(image, sim=True) for image in exitwave])
            templates = np.array([self.filter(template)
                                  for template in templates])
        
        sim = exitwave[:,self.loc[0]:self.loc[0]+h, self.loc[1]:self.loc[1]+w].flatten()
        exp = templates.flatten()
        res = correlation(sim, exp)
        
        # total = match_template(self.exitwave, self.templates)
        print(f'correlation: {res}')
        # self.corrHistory.append(res)
        return res
    

    def displayCoarseSearch(self, filter=None, threshold=None):
        """Displays the result of the coarse search."""
        self.filter = filter
        self.threshold = threshold
        self.thicknessTarget(self.thickness, self)

        total = match_template(self.exitwave, self.templates)

        fig, axes = plt.subplots(
            2, len(self.templates)+1, figsize=(len(self.templates)*4+4, 6.5))
        axes[0, 0].imshow(total[0])
        image = self.exitwave[0, :, :]
        axes[0, 0].plot(self.loc[1], self.loc[0], 'o', markeredgecolor='r',
                        markerfacecolor='none', markersize=10)
        axes[0, 0].set_title(f"correlation map")
        axes[1, 0].imshow(image)
        h, w = self.templates[0].shape
        rect = plt.Rectangle((self.loc[1], self.loc[0]), w, h,
                             edgecolor='white', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].set_title(f"LARBED pattern for 000")
        # axes[0, 0].axis('off')
        # axes[1, 0].axis('off')

        for i, template in enumerate(self.templates):
            image = self.exitwave[i, :, :]
            h, w = template.shape
            crop = image[self.loc[0]:self.loc[0]+h, self.loc[1]:self.loc[1]+w]
            if self.filter:
                crop = self.filter(crop, sim=True)
                template = self.filter(template)
            if self.threshold:
                crop = self.threshold(crop)
                template = self.threshold(template)
            axes[0, i+1].imshow(template)
            axes[0, i+1].axis('off')
            axes[0, i+1].set_title(f"reflection {i+1}")

            axes[1, i+1].imshow(crop)
            axes[1, i+1].axis('off')
            axes[1, i+1].set_title(f"{correlation(crop.flatten(), template.flatten()):.3f}")


class FineOptimize:
    """Class for fine optimization of the structure factor.

    Attributes:
        dinfo (BaseDiffractionInfo): The main diffraction information object.
        thickness (float): The thickness of the sample in Angstroms from dinfo.
        lastParam (SimParams): The last used simulation parameters.
        roi (BaseROI): The region of interest object for refinement.
        refineProbe (bool): Flag to indicate if probe parameters should be refined.
        reflections (List[Tuple[int, int, int]]): List of (h, k, l) Miller indices
            for the reflections being optimized.
        sfMask (np.ndarray): Mask to select which structure factor components to
            optimize.
        noncentro (bool): Flag for non-centrosymmetric structures.
        errorFunc (BaseGOF): The goodness-of-fit function used for optimization.
        symUpdate (bool): Flag to update Ugh matrix with symmetry equivalents.
        history (List): A list to store the optimization history.
    """

    def __init__(self, dinfo: BaseDiffractionInfo, roi: BaseROI, reflections: List[Tuple[int, int, int]], sfMask: np.ndarray, noncentro: bool = False, errorFunc: BaseGOF = None, perturbROI: bool = False, shiftDisk: bool = False, symUpdate = False, probe: Tuple[float, float] = None, refineProbe: bool = True):
        """Initializes the FineOptimize object.

        Args:
            dinfo (BaseDiffractionInfo): Diffraction information object.
            roi (BaseROI): refineROI class object.
            reflections (List[Tuple[int, int, int]]): List of reflection indices 
                to optimize.
            sfMask (np.ndarray): Mask for selecting which components of the 
                structure factor to optimize.
            noncentro (bool, optional): Whether to optimize the phase of the 
                structure factor (for non-centrosymmetric materials). 
                Defaults to False.
            errorFunc (BaseGOF, optional): Error function to use for optimization. 
                Defaults to None.
            perturbROI (bool, optional): Whether to optimize the region of interest. 
                Defaults to False.
            shiftDisk (bool, optional): Whether to optimize the shift of individual 
                disks. Defaults to False.
            symUpdate (bool, optional): Whether to update the Ugh matrix with 
                symmetry equivalent beams. Defaults to False.
            probe (Tuple[float, float], optional): Probe parameters (sigma, gamma).
                Defaults to None.
            refineProbe (bool, optional): Whether to refine probe parameters.
                Defaults to True.
        """
        self.dinfo = dinfo
        self.thickness = dinfo.thickness
        self.lastParam = dinfo.lastParam
        self.roi = roi
        self._probe = probe
        self.refineProbe = refineProbe

        # identify if the optimization is for LARBED or CBED

        self.reflections = reflections
        self.sfMask = sfMask
        self.noncentro = noncentro
        self.errorFunc = errorFunc
        self._perturbROI = perturbROI
        self._shiftDisk = shiftDisk
        self.symUpdate = symUpdate
        self.history = []
        self._nfit = 0

        # calculate the degrees of freedom
        self.calDOF()


    
    @property
    def perturbROI(self):
        """bool: Flag to indicate if ROI parameters (thickness, gl, rotation) should be refined."""
        return self._perturbROI
    
    @perturbROI.setter
    def perturbROI(self, value: bool):
        self._perturbROI = value
        self.calDOF()

    @property
    def probe(self):
        """tuple[float, float] | None: Probe parameters (sigma, gamma) for convolution."""
        return self._probe
    @probe.setter
    def probe(self, value: Tuple[float, float]):
        if value is not None:
            if len(value) != 2:
                raise ValueError(f"probe must be a tuple of length 2, but got {value}")
            self._probe = value
        else:
            self._probe = None
        self.calDOF()

    
    @property
    def shiftDisk(self):
        """bool: Flag to indicate if individual disk shifts should be refined."""
        return self._shiftDisk
    
    @shiftDisk.setter
    def shiftDisk(self, value: bool):
        self._shiftDisk = value
        if value:
            if not hasattr(self.roi, 'diskshift'):
                raise ValueError(
                    "shiftDisk is True, but roi.diskshift is not set. Please set roi.diskshift first.")
        self.calDOF()

    
    def calDOF(self) -> int:
        """Calculates the degrees of freedom for the optimization.

        Returns:
            int: the degree of freedom for the optimization.
        """
        nfit = 1
        if self.perturbROI:
            nfit += 4            
        if self.probe and self.refineProbe:
            nfit += 2
        if self.shiftDisk:
            nfit += self.roi.diskshift.size        

        self.dof = self.roi.npoints-np.sum(self.sfMask) - nfit


    def getx0(self, x0=None) -> None:
        """Gets the initial guess for the optimization.

        Args:
            x0 (np.ndarray, optional): Initial guess for structure factors. 
                Shape (len(reflections), 4) for non-centro, 
                (len(reflections), 2) for centro. Defaults to None.
        """
        if x0 is None:
            if not hasattr(self, 'x0'):
                # get the initial guess for the optimization from dinfo
                if self.noncentro:
                    self.x0 = np.zeros((len(self.reflections), 4))
                    for idx, reflection in enumerate(self.reflections):
                        self.x0[idx, :] = self.dinfo.getSF(reflection)
                else:
                    self.x0 = np.zeros((len(self.reflections), 2))
                    for idx, reflection in enumerate(self.reflections):
                        sf = self.dinfo.getSF(reflection)
                        self.x0[idx, :] = sf[[0, 2]]
        else:
            if self.noncentro:
                if x0.shape != (len(self.reflections), 4):
                    raise ValueError(
                        f"x0 must have shape {(len(self.reflections), 4)}, but got {x0.shape}")
            else:
                if x0.shape != (len(self.reflections), 2):
                    raise ValueError(
                        f"x0 must have shape {(len(self.reflections), 2)}, but got {x0.shape}")
            self.x0 = x0

    def getRange(self, x0Range=None) -> None:
        """Sets the range to normalize the structure factor."""

        if x0Range is None:
            if not hasattr(self, 'x0Range'):
                refSF = lookupSF(self.reflections) if self.noncentro else lookupSF(
                    self.reflections)[:, ::2]

                self.x0Range = (refSF*0.95, refSF*1.05)

        else:
            if x0Range.shape != (2, *self.x0.shape):
                raise ValueError(
                    f"x0Range must have shape {(2, *self.x0.shape)}, but got {x0Range.shape}")
            self.x0Range = x0Range

        print("setting the initial value and normalization range to:")
        if self.noncentro:
            for idx, (sf, reflection) in enumerate(zip(self.x0, self.reflections)):
                print(f"beam: {reflection}")
                print(f"    {'lower':^9}   {'value':^9}   {'upper':^9}")
                print(f"|U|: {self.x0Range[0][idx][0]:.7e}   {sf[0]:.7e}   {self.x0Range[1][idx][0]:.7e}")
                print(f"angle(U): {self.x0Range[0][idx][1]:.7e}   {sf[1]:.7e}   {self.x0Range[1][idx][1]:.7e}")
                print(
                    f"|UA|: {self.x0Range[0][idx][2]:.7e}   {sf[2]:.7e}   {self.x0Range[1][idx][2]:.7e}")
                print(
                    f"angle(UA): {self.x0Range[0][idx][3]:.7e}   {sf[3]:.7e}   {self.x0Range[1][idx][3]:.7e}")
                print('-'*30)


        else:
            for idx, (sf, reflection) in enumerate(zip(self.x0, self.reflections)):
                print(f"beam: {reflection}")
                print(f"    {'lower':^9}   {'value':^9}   {'upper':^9}")
                print(
                    f"U : {self.x0Range[0][idx][0]:.7e}   {sf[0]:.7e}   {self.x0Range[1][idx][0]:.7e}")
                print(
                    f"UA: {self.x0Range[0][idx][1]:.7e}   {sf[1]:.7e}   {self.x0Range[1][idx][1]:.7e}")
                print('-'*30)

    def normalizeX0(self):
        """Normalizes the structure factor and applies the mask."""
        return ((self.x0-self.x0Range[0])/(self.x0Range[1]-self.x0Range[0]))[self.sfMask]

    def denormalizeX0(self, x0):
        """Denormalizes the structure factor and removes the mask."""
        res = np.zeros_like(self.x0)
        res[self.sfMask] =  x0*(self.x0Range[1]-self.x0Range[0])[self.sfMask]+self.x0Range[0][self.sfMask]
        res[~self.sfMask] = self.x0[~self.sfMask]
        return res

    def optimize(self, x0: np.ndarray = None, x0Range: np.ndarray = None) -> None:
        """Runs the fine optimization.

        Args:
            x0 (np.ndarray, optional): Initial guess for structure factors.
                Defaults to None.
            x0Range (np.ndarray, optional): Range for structure factor normalization.
                Defaults to None.
        """
        self.getx0(x0)
        self.getRange(x0Range)
        # TODO: initialize the Ugh matrix??
        # initialize Ugh matrix with strcutre factor ratio recored in dinfo
        # sf, allSymBeam = self.dinfo.getAllSF()
        # update Ugh matrix with the saved structure factor
        # if self.noncentro:
        #     updateSymUgh(allSymBeam, sf, self.dinfo.symmetry)
        # else:
        #     updateSymUgh(allSymBeam, sf[:,::2], self.dinfo.symmetry)

        # visualize the optimization

        self.fig, self.axes = plt.subplots(
            ncols=len(self.roi.templates)+1, figsize=(len(self.roi.templates)*6+2, 5))
        self.expLines = []
        self.simLines = []
        for i in range(self.roi.templates.shape[0]):
            alpha = 1
            # set transparency for the beam not included
            if self.roi.maskConfig is not None and self.roi.maskConfig[0,i] == 0:
                alpha = 0.4
            self.expLines.append(self.axes[i].plot(
                [], label='exp', linestyle=':', alpha=alpha)[0])
            self.simLines.append(self.axes[i].plot([], label='sim', alpha=alpha)[0])
            self.axes[i].set_title(f"{self.roi.gInclude[i]}")
            self.axes[i].set_xlim(0, np.sum(self.roi.regions[:, 3, 0]))
            self.axes[i].set_ylim(
                np.min(self.roi.templates[i])-5, np.max(self.roi.templates[i])+5)
            # if beam not include set alpha to 0.5
            

        self.axes[i].legend()
        self.texts = []
        for i in range(len(self.x0)+1):
            if self.noncentro:
                self.texts.append(self.axes[-1].text(0.1, 0.8 - i*0.3, ""))
            else:
                self.texts.append(self.axes[-1].text(0.1, 0.8 - i*0.1, ""))

        self.axes[-1].axis('off')
        self.axes[-1].set_xlim(0, 1)
        self.axes[-1].set_ylim(0, 1)
        self.fig.canvas.draw()
        self.figBackground = []
        for ax in self.axes:
            self.figBackground.append(self.fig.canvas.copy_from_bbox(ax.bbox))
        plt.show(block=False)

        # start the minimization
        print('start optimization')
        res = minimize(self.SFtarget, self.normalizeX0().flatten(), args=(self,),
                       method=constant['fine']['method'],
                       # bounds=self._boundary,
                       callback=lambda intermediate_result: self.callback(
                           intermediate_result, self),
                       options=constant['fine']['options'])
        self.history.append(res)

        # update the structure factor ratio
        self.x0 = self.denormalizeX0(res.x)

        # update the dinfo
        self.evaluateSF(self.x0)
        self.dinfo.lastParam = self.lastParam
        self.dinfo.gl = self.roi.gl
        self.dinfo.thickness = self.thickness

        # TODO update tilt

        # update refined structure factor to dinfo
        for i, reflection in enumerate(self.reflections):
            if self.noncentro:
                self.dinfo.updateSF(reflection, self.x0[i, :])
            else:
                phase = self.dinfo.getSF(reflection)[[1, 3]]
                self.dinfo.updateSF(reflection, np.array([self.x0[i, 0]*np.cos(np.deg2rad(
                    phase[0])), phase[0], self.x0[i, 1]*np.cos(np.deg2rad(phase[0])), phase[1]]))

    def evaluateSF(self, x0: np.ndarray) -> None:
        """Evaluates the structure factor and solve for the eigenvector/values.

        Args:
            x0 (np.ndarray): Structure factor values.
        """
        # adjust the structure factor
        x0 = x0.reshape(len(self.reflections), -1)
        if self.symUpdate:
            updateSymUgh(self.reflections, x0, self.dinfo.symmetry)
        else:
        # updateUgh((np.array(self.reflections)*-1).tolist(), x0, self.dinfo.beamDict)
            updateUgh(self.reflections, x0, self.dinfo.beamDict)
        # solve the eigen problem
        if self._probe is None:
            self.lastParam = tilt_run(
                self.lastParam, self.roi.simGrid, self.roi.indices, ncores=constant['NCORES'])
        else:
            self.lastParam = tilt_run(
                self.lastParam, self.roi.padSimGrid, self.roi.indices, ncores=constant['NCORES'])

    def evaluateParam(self, thickness=None):
        """Evaluates CBED intensity based on Bloch simulation results and geometry.

        Args:
            thickness (float, optional): Sample thickness. Defaults to None.

        Returns:
            np.ndarray: Simulated CBED pattern.
        """
        if thickness is None:
            thickness = self.thickness

        # calculate intensity from eigen vector/value
        simCBED = LARBED_tilt(self.lastParam, thickness,
                              self.roi.indices.shape[0])  
        
        # convolve with a probe function
        if self._probe is not None:
            if self._probe[0] < 1e-6:
                    self._probe = (100, self._probe[1])
            if self._probe[1] < 1e-6:
                self._probe = (self._probe[0], 100)

            split = (self.roi.padRegions[:, 3, 0]*self.roi.padRegions[:, 3, 1]).astype(np.int32)
            subarray = split_array_by_lengths(simCBED, split, axis=1)
            regionSim = [subarray[i].reshape(simCBED.shape[0], int(self.roi.padRegions[i, 3, 1]), int(
            self.roi.padRegions[i, 3, 0])) for i in range(len(self.roi.padRegions))]
            res = []
            for ith in range(len(self.roi.padRegions)):
                ithregion = []
                for j, sim in enumerate(regionSim[ith]):
                    if self.roi.maskConfig is None or self.roi.maskConfig[ith, j] == 1:  
                        ithregion.append(convolve2d(sim, 
                                                    voigt_2d(np.abs(self._probe[0]), np.abs(self._probe[1])), 
                                                    mode='same', 
                                                    boundary='wrap')[self.roi.padding:-self.roi.padding,self.roi.padding:-self.roi.padding].flatten())
                    else:
                        ithregion.append((sim[self.roi.padding:-self.roi.padding,self.roi.padding:-self.roi.padding].flatten()))
                    
                res.append(np.vstack(ithregion))
            simCBED = np.hstack(res)
                                          
        return simCBED

    def display(self, lines, savedir=None):
        """Displays the optimization result.

        Args:
            lines (List[int]): List of line indices to plot.
            savedir (str, optional): Directory to save the plot. Defaults to None.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Experimental and simulated
                regional patterns.
        """
        if self.noncentro:
            for sf, reflection in zip(self.x0, self.reflections):
                print(f"{reflection}: |U| {sf[0]:.7f} angle(U) {sf[1]:.7f} |UA| {sf[2]:.7f} angle(UA) {sf[3]:.7f}")
        else:
            for sf, reflection in zip(self.x0, self.reflections):
                print(f"{reflection}: U {sf[0]:.7f} UA {sf[1]:.7f}")

        # refresh param
        self.evaluateSF(self.x0)
        simCBED = self.evaluateParam()

        # scale the simulation to the experiment
        if hasattr(self.errorFunc, 'scaling'):
            error = self.errorFunc(simCBED, self.roi.templates, self.roi.mask)
            print(f'{self.errorFunc.name}: {error/self.dof:.5f}', flush=True)
            simCBED = self.errorFunc.scaling(simCBED, self.roi.templates, self.roi.mask)
        else:
            simCBED *= np.max(self.roi.templates)/np.max(simCBED)
            print(f'{self.errorFunc.name}: {self.errorFunc(simCBED, self.roi.templates, self.roi.mask)/self.dof:.5f}', flush=True)

        

        # split intensity for each disk
        split = (self.roi.regions[:, 3, 0] *
                 self.roi.regions[:, 3, 1]).astype(np.int32)

        subarray = split_array_by_lengths(simCBED, split, axis=1)
        regionSim = [subarray[i].reshape(simCBED.shape[0], int(self.roi.regions[i, 3, 1]), int(
            self.roi.regions[i, 3, 0])) for i in range(len(self.roi.regions))]

        subarray = split_array_by_lengths(self.roi.templates, split, axis=1)
        regionExp = [subarray[i].reshape(simCBED.shape[0], int(self.roi.regions[i, 3, 1]), int(
            self.roi.regions[i, 3, 0])) for i in range(len(self.roi.regions))]

        # iterate through the regions, one figure for each region
        for ith in range(len(self.roi.regions)):
            fig, axes = plt.subplots(
                3+len(lines), len(simCBED), figsize=(2+len(self.roi.gInclude)*4, 6+len(lines)*4))
            # iterate through the disks
            for i, (sim, exp) in enumerate(zip(regionSim[ith], regionExp[ith])):
                
                if self.roi.maskConfig is None or self.roi.maskConfig[ith, i] == 1:                        
                    axes[0, i].imshow(exp)
                    axes[0, i].axis('off')
                    axes[0, i].set_title(f"exp {i+1}")
                    axes[1, i].imshow(sim)
                    axes[1, i].axis('off')
                    axes[1, i].set_title(
                    f"sim {i+1}, RI = {np.sum(np.abs(exp-sim))/(np.sum(exp))*100:.2f}%")

                    axes[2, i].imshow((exp-sim)/np.max(exp),
                                    cmap='seismic', vmin=-0.15, vmax=0.15)
                    axes[2, i].axis('off')
                    axes[2, i].set_title(f"diff {i+1}")
                    for j, line in enumerate(lines):
                        if line < exp.shape[0]:
                            axes[3+j, i].plot(sim[line, :], label='sim')
                            axes[3+j, i].plot(exp[line, :], label='exp')
                            axes[3+j, 0].set_ylabel(f"line {line}")
                            axes[3+j, i].plot(exp[line, :] -
                                                sim[line, :], label='diff')
                        else:
                            axes[3+j, i].axis('off')
                else:
                    axes[0, i].axis('off')
                    axes[1, i].axis('off')
                    axes[2, i].axis('off')
                    for j in range(3+len(lines)):
                        axes[j, i].axis('off')
            axes[3, -1].legend()
            if savedir is not None:
                plt.savefig(f"{savedir}/region_{ith+1}.png", dpi=300, bbox_inches='tight')
            
        return regionExp, regionSim

    @staticmethod
    def SFtarget(x0, *args):
        """Error function for structure factor optimization.

        Args:
            x0 (np.ndarray): Structure factor values.
            *args: Variable length argument list, expects a FineOptimize instance.

        Returns:
            float: The optimization error.
        """
        self = args[0]

        self._nfit = 0
        # update the structure factor
        currentX0 = self.denormalizeX0(x0)
        print(f"SF:")
        for beam, cx0 in zip(self.reflections, currentX0):
            print(f"{beam}: {' '.join(f'{value:.7e}' for value in cx0)}")
        self.evaluateSF(currentX0)
        if self._perturbROI:
            if self.refineProbe and self._probe:
                roi_res = minimize(self.ROItarget, (self.thickness, self.roi.gl, self.roi.rotation, self._probe[0],self._probe[1], *self.roi.allshift.flatten(
                )), args=(self,), method=constant['fine_geometry']['method'], options=constant['fine_geometry']['options'])
            else:
                roi_res = minimize(self.ROItarget, (self.thickness, self.roi.gl, self.roi.rotation, *self.roi.allshift.flatten(
                )), args=(self,), method=constant['fine_geometry']['method'], options=constant['fine_geometry']['options'])

            if roi_res.success:
                self.thickness = roi_res.x[0]
                self.roi._gl = roi_res.x[1]
                self.roi._rotation = np.deg2rad(roi_res.x[2])
                if self._probe is not None:
                    self._probe = (roi_res.x[3], roi_res.x[4])
                    self.roi.allshift = roi_res.x[5:].reshape(self.roi.allshift.shape)    
                else:
                    self.roi.allshift = roi_res.x[3:].reshape(self.roi.allshift.shape)           
                # refresh the roi
                self.roi.calculatePixelSize()
                self.roi.updateExpGrid()

                
                print(
                    f'thickness: {self.thickness:.2f} gl: {self.roi.gl:.5f} rotation: {self.roi.rotation:.5f}', flush=True)
                error = roi_res.fun*self.roi.npoints / self.dof
                print(
                    f'corner: {self.roi.allshift[0,0]:.5f} {self.roi.allshift[0,1]:.5f}', flush=True)
                print(
                    f'GOF: {error:.5f}  func eval: {roi_res.nfev}', flush=True)
                print('*'*20, flush=True)
                if self._probe is not None:
                    print(
                        f'sigma: {self._probe[0]} gamma: {self._probe[1]}', flush=True)

            else:
                error = np.inf
                print('roi optimization failed', flush=True)

        # seperate shift for each disk
        if self._shiftDisk:
            self.simCBED = self.evaluateParam()
            shift_res = minimize(self.ShiftTarget, self.roi.diskshift.flatten(), args=(
                self, self.simCBED), method=constant['fine_geometry']['method'], options=constant['fine_geometry']['options'])
            if shift_res.success:
                self.roi.diskshift = shift_res.x.reshape(
                    self.roi.diskshift.shape)
                self.roi.updateExpGrid()
                print(
                    f'diskshift: {self.roi.diskshift}\nGOF: {shift_res.fun}', flush=True)
                error = shift_res.fun
            else:
                print('diskshift optimization failed', flush=True)

        if not self._perturbROI and not self._shiftDisk:
            # evaluate the CBED pattern
            #TODO: optimizie thickness
            simCBED = self.evaluateParam()
            error = self.errorFunc(simCBED, self.roi.templates, self.roi.mask)
            error = error / self.dof
            print(f'gof: {error}', flush=True)

        return error

    @staticmethod
    def ROItarget(x0, *args):
        """Error function for ROI optimization.

        Args:
            x0 (np.ndarray): ROI parameters (thickness, gl, rotation, allshift).
            *args: a FineOptimize instance.

        Returns:
            float: The goodness of fit.
        """
        self = args[0]
        self.roi._gl = x0[1]
        self.roi._rotation = np.deg2rad(x0[2])
        if self._probe and self.refineProbe:
            self._probe = (x0[3], x0[4])
            self.roi.allshift = x0[5:].reshape(self.roi.allshift.shape)
        else:
            self.roi.allshift = x0[3:].reshape(self.roi.allshift.shape)
        # if self.RBED and self.roi.probe is not None:
        #     self.roi.sigma = x0[3]
        #     self.roi.allshift = x0[4:].reshape(self.roi.allshift.shape)
        #     self.roi.gaussuian_filter()
        # else:
        

        # refresh the roi
        self.roi.calculatePixelSize()
        self.roi.updateExpGrid()

        simCBED = self.evaluateParam(thickness=x0[0])
        
        error = self.errorFunc(simCBED, self.roi.templates, self.roi.mask)        

        return error / self.roi.npoints

    @staticmethod
    def ShiftTarget(x0, *args):
        """Error function for disk shift optimization.

        Args:
            x0 (np.ndarray): Disk shift values.
            *args: a FineOptimize instance
                and the simulated CBED pattern.

        Returns:
            float: The optimization error.
        """
        self = args[0]
        simCBED = args[1]
        self.roi.diskshift = x0.reshape(self.roi.diskshift.shape)
        self.roi.updateExpGrid()
        return self.errorFunc(simCBED, self.roi.templates)

    @staticmethod
    def callback(intermediate_result, self=None):
        """Callback function for optimization visualization."""
        self.simCBED = self.evaluateParam()

        x = np.arange(np.sum(self.roi.regions[:, 3, 0]))
        # scale the simulation to the experiment
        simCBED = self.errorFunc.scaling(
            self.simCBED, self.roi.templates, self.roi.mask)

        # split intensity for each disk
        split = (self.roi.regions[:, 3, 0] *
                 self.roi.regions[:, 3, 1]).astype(np.int32)
        subarray = split_array_by_lengths(simCBED, split, axis=1)
        # TODO: add a parameter for adjusting which row to display
        # only display the first row of each region
        regionSim = [subarray[i].reshape(simCBED.shape[0], int(self.roi.regions[i, 3, 1]), int(
            self.roi.regions[i, 3, 0]))[:, 0, :] for i in range(len(self.roi.regions))]
        regionSim = np.concatenate(regionSim, axis=1)
        subarray = split_array_by_lengths(self.roi.templates, split, axis=1)
        regionExp = [subarray[i].reshape(simCBED.shape[0], int(self.roi.regions[i, 3, 1]), int(
            self.roi.regions[i, 3, 0]))[:, 0, :] for i in range(len(self.roi.regions))]
        regionExp = np.concatenate(regionExp, axis=1)

        # iterate through the disks
        for i, (eline, sline) in enumerate(zip(self.expLines, self.simLines)):
            sim = regionSim[i]
            exp = regionExp[i]

            self.fig.canvas.restore_region(self.figBackground[i])
            sline.set_data(x, sim)
            eline.set_data(x, exp)
            self.axes[i].draw_artist(sline)
            self.axes[i].draw_artist(eline)
            self.fig.canvas.blit(self.axes[i].bbox)

        self.fig.canvas.restore_region(self.figBackground[-1])
        # display the structure factor
        currentX0 = self.denormalizeX0(intermediate_result.x)
        for i, reflection in enumerate(self.reflections):
            if self.noncentro:
                self.texts[i].set_text(
                    f"U_amp {reflection} {currentX0[i][0]:.4e}\nU_phase  {reflection} {currentX0[i][1]:.4e}\nUA_amp {reflection} {currentX0[i][2]:.4e}\nUA_phase {reflection} {currentX0[i][3]:.4e}")
            else:
                self.texts[i].set_text(
                    f"U {reflection} {currentX0[i][0]:.4e}\nUA {reflection} {currentX0[i][1]:.4e}")
            self.axes[-1].draw_artist(self.texts[i])
        self.texts[-1].set_text(f"GOF: {intermediate_result.fun:.4f}")
        self.axes[-1].draw_artist(self.texts[-1])
        self.fig.canvas.blit(self.axes[-1].bbox)

        self.fig.canvas.flush_events()

def voigt_2d(sigma, gamma):
    """Generates a 2D Voigt profile.

    Args:
        sigma (float): Standard deviation for the Gaussian component.
        gamma (float): Half-width at half-maximum for the Lorentzian component.

    Returns:
        np.ndarray: A 2D array representing the Voigt profile.
    """
    x = np.linspace(-10, 10, 21)
    y = np.linspace(-10, 10, 21)
    X, Y = np.meshgrid(x, y)
    grid = np.sqrt(X**2 + Y**2)
    z_grid = (grid + 1j * gamma) / (sigma * np.sqrt(2))
    voigt_grid = np.real(wofz(z_grid)) / (sigma * np.sqrt(2 * np.pi))
    return voigt_grid
        # x_0 = 0
        # y_0 = 0

        # z_x = (X - x_0 + 1j * gamma) / (sigma * np.sqrt(2))
        # z_y = (Y - y_0 + 1j * gamma) / (sigma * np.sqrt(2))

        # voigt_x = np.real(wofz(z_x)) / (sigma * np.sqrt(2 * np.pi))
        # voigt_y = np.real(wofz(z_y)) / (sigma * np.sqrt(2 * np.pi))

        # return voigt_x * voigt_y
