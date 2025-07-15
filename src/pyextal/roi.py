from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.transform import SimilarityTransform
from skimage.transform import rotate
from skimage.filters import threshold_otsu, sobel


from pyextal.metric import scale, sumx, angle
from pyextal.callBloch import simulate, bloch, bloch_parse, symmetrizeVHKL
from pyextal.Constants import constant


class ROITYPE(Enum):
    BASE = 0
    CBED = 1
    LARBED = 2
class BaseROI():
    '''
    base class for region of interest
    define simulation sampling and experiment sampling
    also store the experiment interpolation function and intensity
    '''
    roitype = ROITYPE.BASE
    def __init__(self, dinfo, rotation,gx, gInclude):
        '''
        args:
            dinfo: diffraction information  
            rotation: rotation angle in degree            
            gInclude: list of reflections to be included (-200), (000), (200) and (400) 
            sampleRate: subsample rate (sample every sampleRate pixel) for the diffraction pattern          
            gx: systematic direction in the diffraction pattern (horizontal direction)
            
        '''
        self.dinfo = dinfo
        self.dp = self.dinfo.dp
        self._rotation = np.deg2rad(rotation)
        self._gl = self.dinfo.gl
        self.gInclude = gInclude
        self.gx = gx

        # minor shift during optimization
        self.allshift = np.zeros((1, 2), dtype=np.float32)
                
        # initialization of Bloch simulation
        self.initialize()

        # match the diffraction disks and simulation output
        self.matchIndex()

        # create interpolation function for the diffraction pattern
        self.createInterp()

    def __str__(self):
        return f'{self.__class__.__name__}({self.gl} {self.rotation}, {self.gx}, {self.sampleRate})'

    def initialize(self):
        '''
        initialize common block for Bloch simulation
        override this method to add more initialization
        '''
        param, includeBeam = bloch(
            self.dinfo.datpath, self.dinfo.tiltX, self.dinfo.tiltY, HKL=True, ncores= constant['NCORES'])
        self.dinfo.includeBeam = includeBeam
        self.dinfo.lastParam = param
        # symmetrize the manually adjusted structure factor
        symmetrizeVHKL()
        self.calculatePixelSize()
        self.initPixsiz = self.pixsiz

    # match the diffraction disks and simulation output
    def matchIndex(self):
        self.indices = []
        for index in self.gInclude:            
            for i, hkl in enumerate(self.dinfo.lastParam.hklout.T):
                if hkl[0] == index[0] and hkl[1] == index[1] and hkl[2] == index[2]:
                    self.indices.append(i)
                    break
        self.indices = np.array(self.indices)

    def createInterp(self):
        '''
        create intepolation function for the expdp
        override this method to return the interpolation function
        '''
        raise NotImplementedError
    
    def selectROI(self, regions, mask=None, padding=0):
        '''
        select the region of interest       
            
            1-------<-n_12->-------2
            |
            ↑
           n_13
            ↓
            |
            3
        args:
            regions:[[[x1,y1],[x2,y2],[x3,y3],[n_12, n_13]],...], number of points between 1 3 and 1 2
            can have multiple regions, w.r.t the dp center
            mask: mask for roi on different reflections, (n_roi, n_disk), 0 for not included, 1 for included
        '''
        self.regions = regions
        # generate the grid for sampling w.r.t the dp center, ij convention
        self.grid = self.generateGrid(self.regions)

        if padding > 0:
            self.padding = padding
            self.padRegions = regions.copy()
            # add padding to the regions
            for i in range(self.padRegions.shape[0]):
                self.padRegions[i][0] -= padding
                self.padRegions[i][1,0] -= padding
                self.padRegions[i][1,1] += padding
                self.padRegions[i][2,0] += padding
                self.padRegions[i][2,1] -= padding
                self.padRegions[i][3] += padding*2
            self.padGrid = self.generateGrid(self.padRegions)


        # generate the grid for smapling experiment diffraction pattern
        self.updateExpGrid()
        # generate the grid for sampling in simulation
        self.updateSimGrid()

        if mask is None:
            self.mask = None
            self.npoints = int(np.sum(self.regions[:,3,0] * self.regions[:,3,1])*len(self.gInclude))
        elif not regions.shape[0] == mask.shape[0] or not mask.shape[1] == len(self.gInclude):
            raise ValueError('mask shape does not match the regions')
        else:
            # regroup the mask so self.mask is in the format of (n_disk, n_roi_pixel)
            self.mask = []
            for reg in mask.T:
                reflection_mask = []
                for m, include in zip(regions[:,3],reg):
                    reflection_mask += [include,]*int(m[0]*m[1])
                self.mask.append(reflection_mask)
            self.mask = np.array(self.mask, dtype=np.bool_)
            self.npoints = self.mask.size
        self.maskConfig = mask

        

        

    def updateSimGrid(self):
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.simGrid = (self.grid[:,1]*xtilt.reshape((3,1)) - self.grid[:,0]*ytilt.reshape((3,1))).T + self.dinfo.lastParam.tilt0
        self.simGrid = self.simGrid.reshape((-1, 3), order='F')

    def updateExpGrid(self):
        '''
        update the region of interest
        '''
        raise NotImplementedError

    def transformDP(self, coordinates):
        '''
        transform the rotated and shifted diffraction pattern back to the original diffraction pattern
        args:
            coordinates: coordinate in the simulated diffraction pattern (systematic direction is horizontal)
        '''
        raise NotImplementedError

    def calculatePixelSize(self):
        '''
        calculate the pixel size of the diffraction pattern
        '''
        self.pixsiz = self.glen/self._gl
        
        
    def generateGrid(self, regions):
        '''
        generate the grid for sampling experiment diffraction pattern
        '''        
        grids = []
        for region in regions:
            wvector = region[1] - region[0]
            hvector = region[2] - region[0]
            x = np.linspace(0, 1, int(region[3,0]))
            y = np.linspace(0, 1, int(region[3,1]))
            row, col = np.meshgrid(x, y)
            grid = wvector[:, np.newaxis, np.newaxis]*row + hvector[:, np.newaxis, np.newaxis]*col
            grids.append(np.vstack((grid[0].flatten(), grid[1].flatten())).T + region[0])
        return np.concatenate(grids, axis=0)

    # properties required recalculation of pixsize upon change
    @property
    def gl(self):
        return self._gl

    @gl.setter
    def gl(self, value):
        self._gl = value
        self.calculatePixelSize()
        self.updateExpGrid()
    
    # properties require resampling upon change
    @property
    def rotation(self):
        return np.rad2deg(self._rotation)

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.deg2rad(value)
        self.updateExpGrid()

    # read only properties
    @property
    def templates(self):
        return self._templates


    # TODO: think of a way to combine multiple roi, maybe a wrapper class? iterate through multiple roi and call run_tilt for each roi
    def __add__(self, other):
        pass


class CBEDROI(BaseROI):
    '''
    class defining region of interest for CBED
    '''

    roitype = ROITYPE.CBED

    def __init__(self, dinfo, rotation, gx, gInclude, dpCenter, dpSize, gy=None):
        '''
        args:
            dinfo: diffraction information  
            rotation: rotation angle in degree                     
            nthgx: list of reflections to be included (e.g. in .dat, xaxis (100), refining (-200), (000), (200) and (400) : [-2,0,2,4]            
            dpCenter: 000 disk center (row0, col0)
            dpSize: simulated dp size (rowsize, colsize)
            sampleRate: subsample rate (sample every sampleRate pixel) for the diffraction pattern          
            gy: second direction if the diffraction pattern is not systematic
        '''
        self.dpCenter = dpCenter
        self.dpSize = dpSize        
        self.diskshift = np.zeros((len(gInclude), 2), dtype=np.float32)
        
        
        super().__init__(dinfo, rotation,gx, gInclude)
        # decompose 
        self.gCoff = []
        for index in self.gInclude:
            coffa = sumx(self.dinfo.lastParam.gmxr, index, self.gx)/scale(self.dinfo.lastParam.gmxr,self.gx)**2
            if gy is not None:
                coffb = sumx(self.dinfo.lastParam.gmxr, index, self.gy)/scale(self.dinfo.lastParam.gmxr,self.gy)**2
            else:
                coffb = 0
            self.gCoff.append((coffa, coffb))

        # handle none systematic diffraction pattern
        if gy is not None:
            self.gy = gy
            # angle between gx and gy        
            self.xyangle = angle(self.dinfo.lastParam.gmxr, self.gx, self.gy)
            # vector of gy in horizontal diffraction pattern
            self.gyvec = scale(self.dinfo.lastParam.gmxr, self.gy)/scale(
                self.dinfo.lastParam.gmxr, self.gx)*np.array([np.sin(self.xyangle), np.cos(self.fxyangle)])
            self.ratio = scale(self.dinfo.lastParam.gmxr, self.gy)/scale(
                self.dinfo.lastParam.gmxr, self.gx)
        else:
            self.gy = (0,0,0)
            self.gyvec = np.array([0,0])
            self.xyangle = 0
            self.ratio = 0
        

    def initialize(self):
        '''
        preprocess the diffraction patterns

        1. rotate the diffraction pattern
        2. simulate the diffraction pattern (this also initialize common block variables in bloch)
        3. align the diffraction pattern with simulated diffraction pattern 
        4. crop the diffraction pattern if crop is not None
        '''

        self.horizontalDP = rotate(
            self.dp, self.rotation, resize=True, preserve_range=True, mode='edge')
        simShapeRow, simShapeCol = self.dpSize
        expShapeRow, expShapeCol = self.horizontalDP.shape

        tempDisplay = np.array([*self.dpCenter, self._gl, *self.dpSize, 0])
        tempDisplay[3] = max(expShapeRow, simShapeRow)
        tempDisplay[4] = max(expShapeCol, simShapeCol)
        # generate simulated diffraction pattern
        simCBED, param, includeBeam = simulate(
            self.dinfo.datpath, self.dinfo.thickness, tempDisplay, self.dinfo.tiltX, self.dinfo.tiltY, HKL=True, ncores= constant['NCORES'])
        # for debugging
        self.simCBED = simCBED
        # get which beams are included in the simulation
        self.dinfo.includeBeam = includeBeam
        self.dinfo.lastParam = param

        # TODO: some bug here, not sure what
        # align the diffraction pattern with simulated diffraction pattern
        self.horizontalDP = self.horizontalDP[:simCBED.shape[0], :simCBED.shape[1]]
        sobel_dp = sobel(self.horizontalDP)
        t = threshold_otsu(sobel_dp)
        edges = sobel_dp > t
        # hough_radii = np.arange(radius-10, radius+10, 1)
        # hough_res = hough_circle(edges, hough_radii)
        # accums, expx, expy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        sobel_dp = sobel(simCBED)
        t = threshold_otsu(sobel_dp)
        edges_sim = sobel_dp > t
        # edges = canny(simCBED, sigma=3)
        # hough_res = hough_circle(edges, hough_radii)
        # accums, simx, simy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        # shift = np.array([-expy[0]+simy[0], -expx[0]+simx[0]])
        from skimage.registration import phase_cross_correlation
        self.shift, _, _ = phase_cross_correlation(edges_sim, edges, normalization=None)        
        self.horizontalDP = sp.ndimage.shift(self.horizontalDP, shift=self.shift, order=3,
                                             mode='nearest')

        # crop the diffraction pattern to the target size
        self.horizontalDP = self.horizontalDP[:simShapeRow, :simShapeCol]

        fig, axes = plt.subplots(nrows=3, ncols=1)
        axes[0].imshow(self.horizontalDP)

        axes[1].imshow(simCBED[:simShapeRow, :simShapeCol])
        axes[2].imshow(self.horizontalDP/np.max(self.horizontalDP)*np.max(simCBED)-simCBED[:simShapeRow, :simShapeCol])

        # initialize self.pixsiz
        # gg length in Angstrom^-1
        self.glen = scale(self.dinfo.lastParam.gmxr,self.dinfo.lastParam.gg)
        self.calculatePixelSize()
        self.initPixsiz = self.pixsiz


    def createInterp(self):
        row = np.arange(self.dp.shape[0])
        col = np.arange(self.dp.shape[1])
        self.interp = RegularGridInterpolator(
            (row, col), self.dp, method='cubic', bounds_error=False, fill_value=-1)

    def displayROI(self):
        '''
        display the diffraction pattern and the region of interest
        '''
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        # show the diffraction pattern
        self.ax.imshow(self.horizontalDP)
        # show the center of the diffraction pattern
        self.ax.scatter(self.dpCenter[1], self.dpCenter[0], c='white', s=10, marker='+')


        if not hasattr(self, 'regions'):
            self.ax.set_title('No region of interest selected')
            return
        # show the region of interest        
        self.ax.set_title('Selected region of interest')
        for i, region in enumerate(self.regions):
            for j, (coffa,coffb) in enumerate(self.gCoff):
                if self.maskConfig is not None:
                    if self.maskConfig[i,j] == 0:
                        continue               
                
                currCorner = self.dpCenter+region[0] + \
                    self._gl*(np.array([0, coffa]) + self.gyvec*coffb)
                wvector = region[1] - region[0]
                hvector = region[2] - region[0]
                points = np.array([currCorner, currCorner + wvector, currCorner + wvector + hvector, currCorner + hvector])
                points = points[:, ::-1]
                roi = Polygon(points, fill=False, color='white', linestyle='--')
                self.ax.add_patch(roi)

    def transformDP(self, coordinates):
        '''
        transform the rotated and shifted diffraction pattern back to the original diffraction pattern
        args:
            coordinates: coordinate in the simulated diffraction pattern (systematic direction is horizontal)
        
        modified from skimage.transform.rotate source code
        '''
        rows, cols = self.dp.shape[0], self.dp.shape[1]

        # rotation around center

        center = np.array((cols, rows)) / 2. - 0.5

        tform1 = SimilarityTransform(translation=center)
        tform2 = SimilarityTransform(rotation=self._rotation)
        tform3 = SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1

        # determine shape of output image
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform

        # Make sure the transform is exactly affine, to ensure fast warping.
        tform.params[2] = (0, 0, 1)
        coordinates = (coordinates - self.shift)[:, ::-1]
        self.inverse = tform.inverse
        
        return tform(coordinates)[:, ::-1]

    def updateExpGrid(self):
        '''
        sample the orignal diffraction pattern
        '''
        self._templates = []       
        sampleGrid = self.transformDP(self.grid*self.initPixsiz/self.pixsiz + self.dpCenter)
        # potentially can be optimized
        vecgy = rotation(np.array([0, self._gl*self.ratio]), self._rotation+self.xyangle)
        for idx, (coffx, coffy) in enumerate(self.gCoff):
            # calculate g vector and shift
            currGrid = sampleGrid + \
            self._gl*np.array([np.sin(self._rotation), np.cos(self._rotation)])*coffx + \
            vecgy*coffy + \
            self.allshift + \
            self.diskshift[idx]

            self._templates.append(self.interp(
                currGrid))
        self._templates = np.array(self._templates).astype(np.float32)

    def setTilt0(self, tilt, refPoint):
        '''
        set the tilt0 base on given tilt vector and the reference point
        '''
        # not sure if -1 is correct
        print(self.inverse(refPoint[::-1])[0,::-1])
        offset = self.inverse(refPoint[::-1])[0,::-1] + self.shift - (np.array(self.dpCenter))
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.dinfo.lastParam.tilt0 = tilt + offset[0]*ytilt - offset[1]*xtilt

        if hasattr(self, 'simGrid'):
            self.updateSimGrid()
    
    def kt2pixel(self, kt):
        '''
        convert kt to pixel
        args:
            kt: tangential tilt vector
        
        return: pixel coordinate in horizontal diffraction pattern, relative to dpCenter
        '''
        offset = (kt - self.dinfo.lastParam.tilt0)
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        offset_x = np.dot(offset, xtilt) / np.linalg.norm(xtilt)**2
        offset_y = np.dot(offset, ytilt) / np.linalg.norm(ytilt)**2
        # grid = self.transformDP(np.array([-offset_y, offset_x])*self.initPixsiz/self.pixsiz + self.dpCenter)
    
        return np.vstack((-offset_y, offset_x)).T

    def pixel2exp(self, pixel):
        '''
        convert pixel to experiment coordinate
        args:
            pixel: pixel coordinate in horizontal diffraction pattern, relative to dpCenter
        
        return: experiment coordinate in Angstrom^-1
        '''
        return self.transformDP(pixel*self.initPixsiz/self.pixsiz + self.dpCenter)
    
    # for debugging
    def getSampling(self):
        grids = []
        sampleGrid = self.transformDP(self.grid*self.initPixsiz/self.pixsiz + self.dpCenter)
        vecgy = rotation(np.array([0, self._gl*self.ratio]), self._rotation+self.xyangle)
        for idx, (coffx, coffy) in enumerate(self.gCoff):
            # calculate g vector and shift
            currGrid = sampleGrid + \
            self._gl*np.array([np.sin(self._rotation), np.cos(self._rotation)])*coffx + \
            vecgy*coffy + \
            self.allshift + \
            self.diskshift[idx]

            grids.append(currGrid)
        
        return grids


# to be implemented, if same as base just delete the method
class LARBEDROI(BaseROI):
    '''
    class for selecting region of interest for LARBED
    '''
    roitype = ROITYPE.LARBED
    
    def __init__(self, dinfo, rotation, gx, gInclude, probe = None):
        
        self.diskshift = np.zeros((len(gInclude), 2), dtype=np.float32)
        self.indices = []                
        self.sigma = 1        
        
        super().__init__(dinfo, rotation, gx, gInclude)


    def initialize(self):
        
        '''
        preprocess the diffraction patterns       
        - parse the .dat file to initialize the lastParam and includeBeam
        - calibrate the diffraction pattern
        '''

        # self.horizontalDP = rotate(
        #     np.flip(self.dp[self.dp_index],axis=1), self.rotation, resize=True, preserve_range=True, mode='constant', cval=0.0)
        # initialization of Bloch simulation
        param, includeBeam = bloch_parse(self.dinfo.datpath, self.dinfo.tiltX, self.dinfo.tiltY, HKL=True)
        self.dinfo.includeBeam = includeBeam

        self.dinfo.lastParam = param
        self.glen = scale(self.dinfo.lastParam.gmxr,self.dinfo.lastParam.gg)
        self.calculatePixelSize()
        self.initPixsiz = self.pixsiz


    def createInterp(self):
        row = np.arange(self.dp.shape[1])
        col = np.arange(self.dp.shape[2])
        self.interp = []
        self.varInterp = []
        for idx in range(len(self.gInclude)):
            self.interp.append(RegularGridInterpolator(
            (row, col), self.dp[idx], method='cubic', bounds_error=False, fill_value=-1))
            self.varInterp.append(RegularGridInterpolator(
            (row, col), self.varianceMaps[idx], method='cubic', bounds_error=False, fill_value=-1))
        
    def matchIndex(self):
        temp = []
        dpIndex = []
        for index in self.gInclude:
            for i, hkl in enumerate(self.dinfo.lastParam.hklout.T):            
                if hkl[0] == index[0] and hkl[1] == index[1] and hkl[2] == index[2]:
                    temp.append(i)
                    break
            for j, hkl in enumerate(self.dinfo.gindex):
                if hkl[0] == index[0] and hkl[1] == index[1] and hkl[2] == index[2]:
                    dpIndex.append(j)
                    break
        self.dp = self.dp[dpIndex]
        self.varianceMaps = self.dinfo.varianceMaps[dpIndex]
        self.indices = np.array(temp)
    
    def displayROI(self):
        '''
        display the diffraction pattern and the region of interest
        '''
        self.fig, self.ax = plt.subplots(nrows=1, ncols=len(self.gInclude), figsize=(len(self.gInclude)*5, 5))
        # show the diffraction pattern
        if len(self.gInclude) == 1:
            self.ax = [self.ax]
        
        for i, index in enumerate(self.gInclude):
            self.ax[i].imshow(rotate(self.dp[i], self.rotation, resize=True, preserve_range=True, mode='constant', cval=0.0))
        # show the center of the diffraction pattern
            # self.ax[i].scatter(self.dpCenter[1], self.dpCenter[0], c='white', s=10, marker='+')

        if not hasattr(self, 'regions'):
            self.ax[0].set_title('No region of interest selected')
            return
        
        # show the region of interest        
        self.ax[0].set_title('Selected region of interest')
        for i, region in enumerate(self.regions):
            for j in range(len(self.gInclude)):
                if self.maskConfig is not None:
                    if self.maskConfig[i,j] == 0:
                        continue
                currCorner = region[0]
                wvector = region[1] - region[0]
                hvector = region[2] - region[0]
                points = np.array([currCorner, currCorner + wvector, currCorner + wvector + hvector, currCorner + hvector])
                points = points[:, ::-1]
                roi = Polygon(points, fill=False, color='white', linestyle='--')
                self.ax[j].add_patch(roi)

    def transformDP(self, coordinates):
        '''
        transform the rotated and shifted diffraction pattern back to the original diffraction pattern
        args:
            coordinates: coordinate in the simulated diffraction pattern (systematic direction is horizontal)
        
        # modified from skimage.transform.rotate source code
        '''
        rows, cols = self.dp.shape[1], self.dp.shape[2]

        # rotation around center

        center = np.array((cols, rows)) / 2. - 0.5

        tform1 = SimilarityTransform(translation=center)
        tform2 = SimilarityTransform(rotation=self._rotation)
        tform3 = SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1

        # determine shape of output image
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform

        # Make sure the transform is exactly affine, to ensure fast warping.
        tform.params[2] = (0, 0, 1)
        coordinates = (coordinates)[:, ::-1]
        # for debugging
        self.inverse = tform.inverse
        
        return tform(coordinates)[:, ::-1]

    def updateExpGrid(self):
        '''
        sample the orignal diffraction pattern
        '''
        self._templates = []      
        self.expGrid = []
        self.variance = []
        sampleGrid = self.transformDP(self.grid*self.initPixsiz/self.pixsiz )
        # potentially can be optimized
        for idx in range(len(self.gInclude)):
            self.expGrid.append(sampleGrid + self.allshift + self.diskshift[idx])
            self._templates.append(self.interp[idx](self.expGrid[idx]))
            self.variance.append(self.varInterp[idx](self.expGrid[idx]))
        self._templates = np.array(self._templates).astype(np.float32)
        self.variance = np.array(self.variance).astype(np.float32)
    
    def updateSimGrid(self):
        img = rotate(self.dp[0], self.rotation, resize=True, preserve_range=True, mode='constant', cval=0.0)
        self.dpCenter = [img.shape[0]//2, img.shape[1]//2]
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.simGrid = ((self.grid[:,1] - self.dpCenter[1])*xtilt.reshape((3,1)) - (self.grid[:,0]-self.dpCenter[0])*ytilt.reshape((3,1))).T + self.dinfo.lastParam.tilt0
        self.simGrid = self.simGrid.reshape((-1, 3))

        if hasattr(self, 'padGrid'):
            self.padSimGrid = ((self.padGrid[:,1] - self.dpCenter[1])*xtilt.reshape((3,1)) - (self.padGrid[:,0]-self.dpCenter[0])*ytilt.reshape((3,1))).T + self.dinfo.lastParam.tilt0
            self.padSimGrid = self.padSimGrid.reshape((-1, 3))

        

    def setTilt0(self, tilt, refPoint):
        '''
        set the tilt0 base on given tilt vector and the reference point
        '''
        # not sure if -1 is correct
        offset = self.inverse(refPoint[::-1]-1)[0,::-1] 
        ytilt, xtilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.dinfo.lastParam.tilt0 = tilt - offset[1]*xtilt + offset[0]*ytilt
        

        if hasattr(self, 'simGrid'):
            self.updateSimGrid()


    
    # for debugging
    def getSampling(self):
        grids = []
        sampleGrid = self.transformDP(self.grid*self.initPixsiz/self.pixsiz )
        # potentially can be optimized
        for idx in range(len(self.gInclude)):
            # calculate g vector and shift
            currGrid = sampleGrid + self.allshift + self.diskshift[idx]
            grids.append(currGrid)
        return grids


def pix2Tilt(param, pixsiz):
    '''
        convert pixel size to tilt unit
    '''
    xtilt = param.gg / scale(param.gmxr, param.gg) * pixsiz
    ytilt = param.gh / scale(param.gmxr, param.gh) * pixsiz
    return xtilt, ytilt


def split_array_by_lengths(A, B, axis=0):
    # Verify that sum(B) equals len(A)
    if np.sum(B) != A.shape[axis]:
        raise ValueError("The sum of elements in B must equal the length of A.")
    return np.split(A, np.cumsum(B[:-1]), axis=axis)
    

def rotation(vec, theta):
    '''
    rotate a vector by theta
    '''
    rot = np.array([[np.sin(theta), np.cos(theta)], [-np.cos(theta), np.sin(theta)]])
    return np.dot(rot, vec)
