from time import time
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.ndimage import zoom, convolve, gaussian_filter
from scipy.spatial.distance import correlation
from scipy.signal import convolve2d
from scipy.special import wofz
from skimage.feature import match_template
from skimage.transform import rotate
import matplotlib.pyplot as plt

from pyextal.dinfo import BaseDiffractionInfo
from pyextal.roi import BaseROI, split_array_by_lengths, ROITYPE
from pyextal.gof import BaseGOF
from pyextal.callBloch import bloch_run, bloch_parse, LARBED_tilt, LARBED, calibrateLARBED, lookupSF, updateUgh, updateSymUgh, tilt_run, tilt, SimParams
from pyextal.Constants import constant

import pyextal.blochwave as blochwave


class CoarseOptimize:
    '''
        class for coarse optimization of the thickness, orientation and gl
    '''

    def __init__(self, dinfo: BaseDiffractionInfo, roi: BaseROI, searchRadius: float = None, nx: int = None):
        '''
        args:
            dinfo: diffraction information
            roi: refineROI class
            searchRadius: search radius for the optimization, default is None
            nx: number of pixels in radius, default is None

        '''

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

    

    def optimizeOrientationGeometry(self, target, targetValue) -> None:
        self.res = minimize_scalar(target,
                                   args=(self,),
                                   method='brent',
                                   bracket=[targetValue -
                                            5, targetValue+5],
                                   options={'maxiter': 200, 'disp': True, 'xtol': 1e-3})
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
        '''
        optimize the thickness and orientation of experiment pattern
        '''
        self.filter = filter
        self.threshold = threshold
        thickness = self.optimizeOrientationGeometry(self.thicknessTarget, self.thickness)        
        self.dinfo.thickness = thickness
        self.thickness = thickness        
        print(f"thickness: {self.dinfo.thickness}, gl: {self.dinfo.gl}, tiltY: {self.dinfo.tiltY}, tiltX: {self.dinfo.tiltX}", flush=True)

    def optimizeOrientationGL(self, filter=None, threshold=None) -> None:
        '''
        optimize the gl and orientation of experiment pattern
        '''
        self.filter = filter
        self.threshold = threshold
        gl = self.optimizeOrientationGeometry(self.glTarget, self.gl)
        self.dinfo.gl = gl
        self.gl = gl
        self.roi.gl = gl
        print(f"thickness: {self.dinfo.thickness}, gl: {self.dinfo.gl}, tiltY: {self.dinfo.tiltY}, tiltX: {self.dinfo.tiltX}", flush=True)
    
    @staticmethod
    def thicknessTarget(x0, *args):
        '''
        error function for the optimization
        args:
            x0: thickness
            args[0]: CoarseOptimize class
        return:
            error: 1 - max correlation
        '''
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
        '''
        error function for the optimization
        args:
            x0: gl
            args[0]: CoarseOptimize class
        return:
            error: 1 - max correlation
        '''
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
        y0 = blochwave.cryst.dw[:blochwave.cryst.natoms]
        boundary = []
        for i in range(blochwave.cryst.natoms):
            boundary.append((0.0,2.0))
        self.res = minimize(self.correlationTargetDWF, y0, args=(self,),
                        method='Powell',
                        bounds=boundary,
                        callback=self.callbackDWF,
                        options={'maxiter': 10000, 'disp': True, 'fatol': 1e-5, 'xatol':1e-5, 'adaptive':True})   
        self.dw = y0
        blochwave.cryst.dw[:blochwave.cryst.natoms] = y0
        print(
            f"DWF: {self.dw}")
    @staticmethod
    def callbackDWF(y0):
        print(y0)

    @staticmethod
    def correlationTargetDWF(y0, *args):
        '''    
                res = minimize(self.SFtarget, self.normalizeX0().flatten(), args=(self,),
                       method='Nelder-Mead',
                       # bounds=self._boundary,
                       callback=lambda intermediate_result: self.callback(
                           intermediate_result, self),
                       options={'maxiter': 10000, 'disp': True, 'fatol': 1e-4, 'xatol':1e-3, 'adaptive':True})    
        error function for the optimization
        args:
            x0: thickness and gl
            args[0]: CoarseOptimize class
        return:
            error: 1 - max correlation
        '''
        
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
        y0 = []
        for i in range(blochwave.cryst.natoms):
            y0.append(blochwave.cryst.atpar[0][i])
            y0.append(blochwave.cryst.atpar[1][i])
            y0.append(blochwave.cryst.atpar[2][i])
        boundary = []
        for i in range(len(y0)):
            boundary.append((y0[i]-0.005,y0[i]+0.005))
        self.res = minimize(self.correlationTargetXYZ, y0, args=(self,),
                        method='Nelder-Mead',
                        bounds=boundary,
                        #callback=lambda intermediate_result: self.callback(
                        #intermediate_result, self),
                        options={'maxiter': 10000, 'disp': True, 'fatol': 1e-5, 'xatol':1e-5, 'adaptive':True})   
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
        '''    
                res = minimize(self.SFtarget, self.normalizeX0().flatten(), args=(self,),
                       method='Nelder-Mead',
                       # bounds=self._boundary,
                       callback=lambda intermediate_result: self.callback(
                           intermediate_result, self),
                       options={'maxiter': 10000, 'disp': True, 'fatol': 1e-4, 'xatol':1e-3, 'adaptive':True})    
        error function for the optimization
        args:
            y0: xyz coordinates for each atom
            args[0]: CoarseOptimize class
        return:
            error: 1 - max correlation
        '''
        
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
        '''    
                res = minimize(self.SFtarget, self.normalizeX0().flatten(), args=(self,),
                       method='Nelder-Mead',
                       # bounds=self._boundary,
                       callback=lambda intermediate_result: self.callback(
                           intermediate_result, self),
                       options={'maxiter': 10000, 'disp': True, 'fatol': 1e-4, 'xatol':1e-3, 'adaptive':True})    
        error function for the optimization
        args:
            y0: xyz coordinates for each atom
            args[0]: CoarseOptimize class
        return:
            error: 1 - max correlation
        '''
        
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
    
    def optimzeHV(self, filter=None, threshold=None) -> None:
        '''
        optimize the HV
        '''
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
        '''
        error function for the optimization
        args:
            x0: HV
            args[0]: CoarseOptimize class
        return:
            error: 1 - max correlation
        '''
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
        '''
        display the result of the coarse search
        '''
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
    '''
    class for fine optimization of the structure factor
    '''

    def __init__(self, dinfo: BaseDiffractionInfo, roi: BaseROI, reflections: List[Tuple[int, int, int]], sfMask: np.ndarray[np.bool_], noncentro: bool = False, errorFunc: BaseGOF = None, perturbROI: bool = False, shiftDisk: bool = False, symUpdate = False, probe=None):
        '''
        args:
            dinfo: diffraction information
            roi: refineROI class
            reflections: list of of tuples, reflection indices to optimize the structure factor
            sfMask: mask for selecting which component of the structure factor to optimize
            noncentro: whether to optimize the phase of the structure factor (noncentrosymmetric material->True)
            errorFunc: error function to use for the optimization
            perturbROI: whether to optimize the region of interest
            shiftDisk: whether to optimize the shift of the individual disk
            symUpdate: whether to update the Ugh matrix with the symmetry equivalent beams            
        '''
        self.dinfo = dinfo
        self.thickness = dinfo.thickness
        self.lastParam = dinfo.lastParam
        self.roi = roi
        self.probe = probe

        # identify if the optimization is for LARBED or CBED
        if self.roi.roitype is ROITYPE.LARBED:
            self.RBED = True
        elif self.roi.roitype is ROITYPE.CBED:
            self.RBED = False
        else:
            raise ValueError(f'ROI type {self.roi.roitype} is not supported')
        self.reflections = reflections
        self.sfMask = sfMask
        self.noncentro = noncentro
        self.errorFunc = errorFunc
        self.peturbROI = perturbROI
        self.shiftDisk = shiftDisk
        self.symUpdate = symUpdate
        self.history = []
        self._nfit = 0

    def getx0(self, x0=None) -> None:
        '''
        get the initial guess for the optimization
        args:
            x0: np.ndarray (len(reflections), 4) or (len(reflections), 2)
        '''
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
        '''
            set the range to normalize the structure factor
        '''

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
        for idx, (sf, reflection) in enumerate(zip(self.x0, self.reflections)):
            print(f"beam: {reflection}")
            print(f"    {'lower':^9}   {'value':^9}   {'upper':^9}")
            print(
                f"U : {self.x0Range[0][idx][0]:.7e}   {sf[0]:.7e}   {self.x0Range[1][idx][0]:.7e}")
            print(
                f"UA: {self.x0Range[0][idx][1]:.7e}   {sf[1]:.7e}   {self.x0Range[1][idx][1]:.7e}")
            print('-'*30)

    def normalizeX0(self):
        '''
        normalize the structure factor
        apply the mask to the structure factor
        '''
        print(f'normal:{self.x0}')        
        return ((self.x0-self.x0Range[0])/(self.x0Range[1]-self.x0Range[0]))[self.sfMask]

    def denormalizeX0(self, x0):
        '''
        denormalize the structure factor
        remove the mask from the structure factor
        '''
        res = np.zeros_like(self.x0)
        res[self.sfMask] =  x0*(self.x0Range[1]-self.x0Range[0])[self.sfMask]+self.x0Range[0][self.sfMask]
        res[~self.sfMask] = self.x0[~self.sfMask]
        return res

    def optimize(self, x0: np.ndarray[np.float32] = None, x0Range: np.ndarray[np.float32] = None) -> None:
        '''
        args:
            x0: np.ndarray (len(reflections), 4) or (len(reflections), 2)
        '''
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
                       method='Nelder-Mead',
                       # bounds=self._boundary,
                       callback=lambda intermediate_result: self.callback(
                           intermediate_result, self),
                       options={'maxiter': 1000, 'disp': True, 'fatol': 1e-4, 'xatol':1e-4, 'adaptive':True})
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

    def evaluateSF(self, x0: np.ndarray[np.float32]) -> None:
        '''
        args:
            x0: structure factor ratio corresponding to the reflections
        '''
        # adjust the structure factor
        x0 = x0.reshape(len(self.reflections), -1)
        if self.symUpdate:
            updateSymUgh(self.reflections, x0, self.dinfo.symmetry)
        else:
        # updateUgh((np.array(self.reflections)*-1).tolist(), x0, self.dinfo.beamDict)
            updateUgh(self.reflections, x0, self.dinfo.beamDict)
        # solve the eigen problem
        if self.probe is None:
            self.lastParam = tilt_run(
                self.lastParam, self.roi.simGrid, self.roi.indices, ncores=constant['NCORES'])
        else:
            self.lastParam = tilt_run(
                self.lastParam, self.roi.padSimGrid, self.roi.indices, ncores=constant['NCORES'])
    # def evaluateSFduple(self, y0: np.ndarray[np.float32]) -> None:
    #     '''
    #     args:
    #         x0: structure factor ratio corresponding to the reflections
    #     '''
    #     # adjust the structure factor
    #     self.x0[self.refl_track] = y0[0]
    #     self.x0[self.refl_track] = y0[1]
    #     x0 = self.x0.reshape(len(self.reflections), -1)

    #     if self.RBED and self.noncentro == False:
    #         updateUgh(self.reflections, x0, self.dinfo.beamDict)
    #         updateUgh((np.array(self.reflections)*-1).tolist(),
    #                   x0, self.dinfo.beamDict)
    #     else:
    #         updateUgh(self.reflections, x0, self.dinfo.beamDict)
    #     # solve the eigen problem
    #     self.lastParam = tilt_run(
    #         self.lastParam, self.roi.tiltGrid, self.roi.indices, ncores=constant['NCORES'])

    def evaluateParam(self, thickness=None):
        '''
        evaluate CBED intensity base on bloch simulation results and geometry parameters
        '''
        if thickness is None:
            thickness = self.thickness

        # calculate intensity from eigen vector/value
        simCBED = LARBED_tilt(self.lastParam, thickness,
                              self.roi.indices.shape[0])  
        
        # convolve with a probe function
        if self.probe is not None:
            if self.probe[0] < 1e-6:
                    self.probe = (100, self.probe[1])
            if self.probe[1] < 1e-6:
                self.probe = (self.probe[0], 100)

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
                                                    voigt_2d(np.abs(self.probe[0]), np.abs(self.probe[1])), 
                                                    mode='same', 
                                                    boundary='wrap')[self.roi.padding:-self.roi.padding,self.roi.padding:-self.roi.padding].flatten())
                    else:
                        ithregion.append((sim[self.roi.padding:-self.roi.padding,self.roi.padding:-self.roi.padding].flatten()))
                    
                res.append(np.vstack(ithregion))
            simCBED = np.hstack(res)
                                          
        return simCBED

    def display(self, lines, savedir=None):
        '''
        display the result
        '''
        for sf, reflection in zip(self.x0, self.reflections):
            print(f"{reflection}: U {sf[0]:.7f} UA {sf[1]:.7f}")

        # refresh param
        self.evaluateSF(self.x0)
        simCBED = self.evaluateParam()

        # scale the simulation to the experiment
        if hasattr(self.errorFunc, 'scaling'):
            error = self.errorFunc(simCBED, self.roi.templates, self.roi.mask)
            print(f'{self.errorFunc.name}: {error/(self.roi.npoints-np.sum(self.sfMask)-self._nfit):.5f}', flush=True)
            simCBED = self.errorFunc.scaling(simCBED, self.roi.templates, self.roi.mask)
        else:
            simCBED *= np.max(self.roi.templates)/np.max(simCBED)
            print(f'{self.errorFunc.name}: {self.errorFunc(simCBED, self.roi.templates, self.roi.mask)/(self.roi.npoints-np.sum(self.sfMask)-self._nfit):.5f}', flush=True)

        

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
        '''
        error function for the optimization
        args:
            x0: structure factor ratio
            args[0]: FineOptimize class
        return:
            error: error
        '''
        self = args[0]

        self._nfit = 0
        # update the structure factor
        currentX0 = self.denormalizeX0(x0)
        print(f"SF:")
        for beam, cx0 in zip(self.reflections, currentX0):
            print(f"{beam}: {' '.join(f'{value:.7e}' for value in cx0)}")
        self.evaluateSF(currentX0)
        if self.peturbROI:
            self._nfit = 5
            if self.probe is not None:
                roi_res = minimize(self.ROItarget, (self.thickness, self.roi.gl, self.roi.rotation, self.probe[0],self.probe[1], *self.roi.allshift.flatten(
                )), args=(self,), method='Powell', options={'maxiter': 1000, 'disp': False, 'xtol': 1e-1})
            else:
                roi_res = minimize(self.ROItarget, (self.thickness, self.roi.gl, self.roi.rotation, *self.roi.allshift.flatten(
                )), args=(self,), method='Powell', options={'maxiter': 1000, 'disp': False, 'xtol': 1e-1})

            if roi_res.success:
                self.thickness = roi_res.x[0]
                self.roi._gl = roi_res.x[1]
                self.roi._rotation = np.deg2rad(roi_res.x[2])
                if self.probe is not None:
                    self.probe = (roi_res.x[3], roi_res.x[4])
                    self.roi.allshift = roi_res.x[5:].reshape(self.roi.allshift.shape)    
                else:
                    self.roi.allshift = roi_res.x[3:].reshape(self.roi.allshift.shape)           
                # refresh the roi
                self.roi.calculatePixelSize()
                self.roi.updateExpGrid()

                
                print(
                    f'thickness: {self.thickness:.2f} gl: {self.roi.gl:.5f} rotation: {self.roi.rotation:.5f}', flush=True)
                error = roi_res.fun*self.roi.npoints / (self.roi.npoints-self._nfit-np.sum(self.sfMask))
                print(
                    f'corner: {self.roi.allshift[0,0]:.5f} {self.roi.allshift[0,1]:.5f}', flush=True)
                print(
                    f'GOF: {error:.5f}  func eval: {roi_res.nfev}', flush=True)
                print(
                    f'Error: {roi_res.fun}  func eval: {roi_res.nfev}', flush=True)
                print('*'*20, flush=True)
                if self.probe is not None:
                    print(
                        f'sigma: {self.probe[0]} gamma: {self.probe[1]}', flush=True)

            else:
                error = np.inf
                print('roi optimization failed', flush=True)

        # seperate shift for each disk
        if self.shiftDisk:
            self._nfit = 6
            self.simCBED = self.evaluateParam()
            shift_res = minimize(self.ShiftTarget, self.roi.diskshift.flatten(), args=(
                self, self.simCBED), method='Powell', options={'maxiter': 1000, 'disp': False, 'xtol': 1e-1})
            if shift_res.success:
                self.roi.diskshift = shift_res.x.reshape(
                    self.roi.diskshift.shape)
                self.roi.updateExpGrid()
                print(
                    f'diskshift: {self.roi.diskshift}\nGOF: {shift_res.fun}', flush=True)
                error = shift_res.fun
            else:
                print('diskshift optimization failed', flush=True)

        if not self.peturbROI and not self.shiftDisk:
            # evaluate the CBED pattern
            simCBED = self.evaluateParam()
            error = self.errorFunc(simCBED, self.roi.templates, self.roi.mask)
            error = error / (self.roi.npoints-np.sum(self.sfMask))
            print(f'gof: {error}', flush=True)

        return error

    @staticmethod
    def ROItarget(x0, *args):
        '''
            error function for roi optimization
            args:
                x0: thickness, gl, rotation, allshift
                args[0]: FineOptimize class
            return:
                error: error
        '''
        self = args[0]
        self.roi._gl = x0[1]
        self.roi._rotation = np.deg2rad(x0[2])
        if self.probe is not None:
            self.probe = (x0[3], x0[4])
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
        # if error <= 0:
        #     error = 1000000000
        # if np.any(simCBED < 0):
        #     error = 1000000000
        # if np.any(self.roi.templates < 0):
        #     error = 1000000000

        return error / self.roi.npoints

    @staticmethod
    def ShiftTarget(x0, *args):
        '''
            error function for disk shift optimization
            args:
                x0: diskshift
                args[0]: FineOptimize class
            return:
                error: error
        '''
        self = args[0]
        simCBED = args[1]
        self.roi.diskshift = x0.reshape(self.roi.diskshift.shape)
        self.roi.updateExpGrid()
        return self.errorFunc(simCBED, self.roi.templates)

    @staticmethod
    def callback(intermediate_result, self=None):
        '''
        callback function for the optimization, for visualization
        '''
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
    """
    Generates a 2D Voigt profile.

    Parameters:
    x, y: 2D arrays of coordinates
    x_0, y_0: Center coordinates
    amplitude: Peak intensity
    sigma_x, sigma_y: Standard deviations for Gaussian component in x and y
    gamma_x, gamma_y: Half-width at half-maximum for Lorentzian component in x and y

    Returns:
    A 2D array representing the Voigt profile
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
