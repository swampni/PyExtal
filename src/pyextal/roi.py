"""Region of Interest (ROI) Management.

This module provides classes for defining, managing, and sampling Regions of
Interest (ROIs) from experimental and simulated diffraction patterns. It includes
base classes and specific implementations for Convergent Beam Electron Diffraction
(CBED) and Large Angle Rocking Beam Electron Diffraction (LARBED).

The core functionalities include:
-   Defining ROIs with geometric shapes.
-   Generating sampling grids for both simulation and experiment.
-   Handling coordinate transformations (rotation, scaling, shifting).
-   Creating interpolation functions for experimental data.
-   Extracting intensity templates from experimental patterns.
"""
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
from pyextal.callBloch import simulate, bloch_parse
from pyextal.Constants import constant


class ROITYPE(Enum):
    """Enumeration for the type of Region of Interest."""
    BASE = 0
    CBED = 1
    LARBED = 2
class BaseROI():
    """Base class for defining and managing a Region of Interest (ROI).

    This class handles the common functionalities for defining sampling grids for
    both simulation and experiment, storing experimental intensity data, and
    managing coordinate transformations. It is intended to be subclassed for
    specific diffraction techniques like CBED or LARBED.

    Attributes:
        dinfo (BaseDiffractionInfo): The main diffraction information object.
        dp (np.ndarray): The experimental diffraction pattern.
        rotation (float): The rotation angle in degrees.
        gl (float): The geometric scaling factor.
        gInclude (list): A list of Miller indices (h, k, l) for the reflections
            to be included in the ROI.
        gx (np.ndarray): The vector defining the primary (horizontal) systematic
            row direction.
        allshift (np.ndarray): A global shift applied to all ROIs.
        indices (np.ndarray): An array of indices mapping the `gInclude` reflections
            to the output of the Bloch wave simulation.
        pixsiz (float): The pixel size in the simulation space.
        initPixsiz (float): The initial pixel size, stored for scaling calculations.
    """
    roitype = ROITYPE.BASE
    def __init__(self, dinfo, rotation,gx, gInclude):
        """Initializes the BaseROI object.

        Args:
            dinfo (BaseDiffractionInfo): The main diffraction information object.
            rotation (float): The rotation angle in degrees.
            gx (np.ndarray): The vector defining the primary (horizontal) systematic
                row direction.
            gInclude (list): A list of Miller indices (h, k, l) for the reflections
                to be included in the ROI.
        """
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
        """Returns a string representation of the ROI object."""
        return f'{self.__class__.__name__}({self.gl} {self.rotation}, {self.gx}, {self.sampleRate})'

    def initialize(self):
        """Initializes the Bloch simulation and calculates initial pixel size.

        This method sets up the fortran module for the Bloch simulation, stores the
        resulting parameters in the `dinfo` object, and calculates the initial
        pixel size based on the simulation geometry. It should be overridden by
        subclasses to add technique-specific initializations.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # match the diffraction disks and simulation output
    def matchIndex(self):
        """Matches the included reflections to the simulation output indices."""
        self.indices = []
        for index in self.gInclude:            
            for i, hkl in enumerate(self.dinfo.lastParam.hklout.T):
                if hkl[0] == index[0] and hkl[1] == index[1] and hkl[2] == index[2]:
                    self.indices.append(i)
                    break
        self.indices = np.array(self.indices)

    def createInterp(self):
        """Creates an interpolation function for the experimental diffraction pattern.

        This method must be implemented by subclasses to handle the specific data
        format of the diffraction experiment (e.g., a single 2D pattern for CBED,
        or a stack of 2D patterns for LARBED).
        """
        raise NotImplementedError
    
    def selectROI(self, regions, mask=None, padding=0):
        """Selects and configures the regions of interest for refinement.

        This method defines the geometric regions to be sampled, generates the
        corresponding sampling grids, and sets up masks to include or exclude
        specific disks within each region.

        The region is defined by three corner points (1, 2, 3) and the number of
        sampling points along the vectors 1->2 and 1->3.

        ::

            1-------(n_12 points)-------2
            |
            (n_13 points)
            |
            3

        Args:
            regions (np.ndarray): An array of shape `(n_regions, 4, 2)` where each
                row defines a region. The format is `[[x1,y1], [x2,y2], [x3,y3], [n_12, n_13]]`.
                Coordinates are relative to the diffraction pattern center.
            mask (np.ndarray, optional): A boolean array of shape `(n_regions, n_disks)`
                where a value of 1 includes the disk in the refinement and 0 excludes
                it. If None, all disks in all regions are included. Defaults to None.
            padding (int, optional): Number of pixels to pad around the ROI, used
                for probe convolution. Defaults to 0.
        """
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
        """Updates the simulation grid based on the current geometry.

        This method calculates the required tilt angles for each point in the
        sampling grid and prepares the `simGrid` attribute for the Bloch wave
        simulation.
        """
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.simGrid = (self.grid[:,1]*xtilt.reshape((3,1)) - self.grid[:,0]*ytilt.reshape((3,1))).T + self.dinfo.lastParam.tilt0
        self.simGrid = self.simGrid.reshape((-1, 3), order='F')

    def updateExpGrid(self):
        """Updates the experimental sampling grid and extracts intensity templates.

        This method must be implemented by subclasses. It should calculate the
        coordinates for sampling the experimental data based on the current
        rotation, scaling, and shift parameters, and then use the interpolation
        function to extract the intensity values (`templates`).
        """
        raise NotImplementedError

    def transformDP(self, coordinates):
        """Transforms coordinates from the ROI frame to the original DP frame.

        This method must be implemented by subclasses. It should account for the
        rotation and shifts applied to the diffraction pattern during preprocessing.

        Args:
            coordinates (np.ndarray): An array of (row, col) coordinates in the
                (potentially rotated and shifted) ROI frame.
        """
        raise NotImplementedError

    def calculatePixelSize(self):
        """Calculates the pixel size based on the geometric scaling factor `gl`."""
        self.pixsiz = self.glen/self._gl
        
        
    def generateGrid(self, regions):
        """Generates a sampling grid from a set of region definitions.

        For each region, it creates a grid of points by linearly interpolating
        between the corner points.

        Args:
            regions (np.ndarray): The region definitions, as described in `selectROI`.

        Returns:
            np.ndarray: A concatenated array of (row, col) coordinates for all
            sampling points in all regions.
        """        
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
        """float: The geometric scaling factor. Recalculates grids upon modification."""
        return self._gl

    @gl.setter
    def gl(self, value):
        self._gl = value
        self.calculatePixelSize()
        self.updateExpGrid()
    
    # properties require resampling upon change
    @property
    def rotation(self):
        """float: The rotation angle in degrees. Recalculates grids upon modification."""
        return np.rad2deg(self._rotation)

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.deg2rad(value)
        self.updateExpGrid()

    # read only properties
    @property
    def templates(self):
        """np.ndarray: The extracted experimental intensity templates for the ROIs."""
        return self._templates


    # TODO: think of a way to combine multiple roi, maybe a wrapper class? iterate through multiple roi and call run_tilt for each roi
    def __add__(self, other):
        """Adds two ROI objects together. Not yet implemented."""
        pass


class CBEDROI(BaseROI):
    """A Region of Interest (ROI) class specifically for CBED patterns.

    This class extends `BaseROI` to handle the specific geometry and data
    associated with a single, large Convergent Beam Electron Diffraction pattern.
    It manages the alignment of the experimental pattern to a simulation and
    defines the positions of diffraction disks based on crystallographic vectors.

    Attributes:
        dpCenter (tuple): The (row, col) coordinates of the (000) disk center.
        dpSize (tuple): The (rows, cols) size of the simulation output.
        diskshift (np.ndarray): An array of (row, col) shifts applied individually
            to each diffraction disk.
        gCoff (list): A list of tuples containing the projection coefficients of
            each `gInclude` vector onto the `gx` and `gy` axes.
    """

    roitype = ROITYPE.CBED

    def __init__(self, dinfo, rotation, gx, gInclude, dpCenter, dpSize, gy=None):
        """Initializes the CBEDROI object.

        Args:
            dinfo (BaseDiffractionInfo): The main diffraction information object.
            rotation (float): The rotation angle in degrees.
            gx (np.ndarray): The vector defining the primary (horizontal) systematic
                row direction.
            gInclude (list): A list of Miller indices (h, k, l) for the reflections
                to be included in the ROI.
            dpCenter (tuple): The (row, col) coordinates of the (000) disk center.
            dpSize (tuple): The (rows, cols) size for the simulation output.
            gy (np.ndarray, optional): The vector for the secondary (vertical)
                direction, for non-systematic row cases. Defaults to None.
        """
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
        """Pre-processes the CBED pattern and initializes the simulation.

        This method performs the following steps:
        1.  Rotates the experimental DP to align the systematic row horizontally.
        2.  Runs an initial Bloch simulation to get the geometry.
        3.  Aligns the experimental DP to the simulated DP using phase cross-correlation
        on their Sobel-filtered edge maps.
        4.  Crops the aligned experimental DP to match the simulation size.
        5.  Calculates the initial pixel size.
        """

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
        """Creates a 2D interpolation function for the entire CBED pattern."""
        row = np.arange(self.dp.shape[0])
        col = np.arange(self.dp.shape[1])
        self.interp = RegularGridInterpolator(
            (row, col), self.dp, method='cubic', bounds_error=False, fill_value=-1)

    def displayROI(self):
        """Displays the pre-processed CBED pattern and the selected ROIs.

        Overlays the defined ROI polygons on the horizontally-aligned experimental
        diffraction pattern.
        """
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
        """Transforms ROI coordinates back to the original, un-rotated DP frame.

        This method calculates the inverse transformation, accounting for the
        rotation and phase-correlation shift applied during initialization.

        Args:
            coordinates (np.ndarray): An array of (row, col) coordinates in the
                horizontally-aligned ROI frame.

        Returns:
            np.ndarray: The corresponding (row, col) coordinates in the original
            diffraction pattern.
        """
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
        """Samples the experimental DP to generate intensity templates.

        This method calculates the final sampling coordinates for each disk by
        applying the geometric scaling (`gl`), rotation, global shift (`allshift`),
        and individual disk shifts (`diskshift`). It then uses the interpolation
        function to extract the intensity values.
        """
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
        """Sets the central tilt of the simulation based on a reference point.

        This allows re-centering the simulation's tilt space (`tilt0`) to a
        specific feature (e.g., a zone axis) identified at `refPoint` in the
        experimental pattern.

        Args:
            tilt (np.ndarray): The new tilt vector to be set as the center.
            refPoint (tuple): The (row, col) coordinates in the horizontal DP
                that correspond to the new `tilt`.
        """
        # not sure if -1 is correct
        print(self.inverse(refPoint[::-1])[0,::-1])
        offset = self.inverse(refPoint[::-1])[0,::-1] + self.shift - (np.array(self.dpCenter))
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.dinfo.lastParam.tilt0 = tilt + offset[0]*ytilt - offset[1]*xtilt

        if hasattr(self, 'simGrid'):
            self.updateSimGrid()
    
    def kt2pixel(self, kt):
        """Converts a tangential tilt vector (kt) to pixel coordinates.

        Args:
            kt (np.ndarray): A tilt vector in the simulation's tangential plane.

        Returns:
            np.ndarray: The corresponding (row, col) pixel coordinates relative
            to the `dpCenter` in the horizontally-aligned DP.
        """
        offset = (kt - self.dinfo.lastParam.tilt0)
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        offset_x = np.dot(offset, xtilt) / np.linalg.norm(xtilt)**2
        offset_y = np.dot(offset, ytilt) / np.linalg.norm(ytilt)**2
        # grid = self.transformDP(np.array([-offset_y, offset_x])*self.initPixsiz/self.pixsiz + self.dpCenter)
    
        return np.vstack((-offset_y, offset_x)).T

    def pixel2exp(self, pixel):
        """Converts pixel coordinates to experimental coordinates.

        Args:
            pixel (np.ndarray): (row, col) pixel coordinates in the horizontal
                diffraction pattern, relative to `dpCenter`.

        Returns:
            np.ndarray: The corresponding coordinates in the original experimental
            diffraction pattern space.
        """
        return self.transformDP(pixel*self.initPixsiz/self.pixsiz + self.dpCenter)
    
    # for debugging
    def getSampling(self):
        """Returns the raw sampling coordinates for debugging purposes."""
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
    """A Region of Interest (ROI) class specifically for LARBED patterns.

    This class extends `BaseROI` to handle LARBED data, which consists of a
    stack of images, one for each diffraction disk. It creates a separate
2D
    interpolation function for each disk's image and its corresponding variance map.

    Attributes:
        diskshift (np.ndarray): An array of (row, col) shifts applied individually
            to each diffraction disk.
        indices (np.ndarray): An array of indices mapping the `gInclude` reflections
            to the output of the Bloch wave simulation.
        varianceMaps (np.ndarray): A stack of variance maps corresponding to the
            diffraction pattern images.
        interp (list): A list of interpolation functions, one for each disk image.
        varInterp (list): A list of interpolation functions for the variance maps.
    """
    roitype = ROITYPE.LARBED
    
    def __init__(self, dinfo, rotation, gx, gInclude, probe = None):
        """Initializes the LARBEDROI object.

        Args:
            dinfo (BaseDiffractionInfo): The main diffraction information object,
                which must be a `LARBEDDiffractionInfo` instance.
            rotation (float): The rotation angle in degrees.
            gx (np.ndarray): The vector defining the primary (horizontal) systematic
                row direction.
            gInclude (list): A list of Miller indices (h, k, l) for the reflections
                to be included in the ROI.
            probe (any, optional): Probe parameters. Currently not used.
                Defaults to None.
        """
        
        self.diskshift = np.zeros((len(gInclude), 2), dtype=np.float32)
        self.indices = []                
        
        super().__init__(dinfo, rotation, gx, gInclude)


    def initialize(self):
        """Initializes the simulation parameters from the LARBED `.dat` file."""
        
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
        """Creates interpolation functions for each disk image and variance map."""
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
        """Matches `gInclude` reflections to simulation and experimental data indices.

        This method maps the desired `gInclude` reflections to both the output
        order of the Bloch simulation (`self.indices`) and the order of the images
        in the experimental data stack (`dpIndex`), then reorders the `dp` and
        `varianceMaps` arrays accordingly.
        """
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
        """Displays the LARBED disk images and the selected ROIs."""
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
        """Transforms ROI coordinates back to the original, un-rotated DP frame.

        Args:
            coordinates (np.ndarray): An array of (row, col) coordinates in the
            ROI frame.

        Returns:
            np.ndarray: The corresponding (row, col) coordinates in the original
            diffraction pattern image frame.
        """
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
        """Samples the LARBED images to generate intensity and variance templates."""
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
        """Updates the simulation grid based on the LARBED geometry.

        For LARBED, the center of the rocking pattern is assumed to be the center
        of the image. This method calculates the tilt vectors corresponding to
        each pixel in the ROI relative to this center.
        """
        img = rotate(self.dp[0], self.rotation, resize=True, preserve_range=True, mode='constant', cval=0.0)
        self.dpCenter = [img.shape[0]//2, img.shape[1]//2]
        xtilt, ytilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.simGrid = ((self.grid[:,1] - self.dpCenter[1])*xtilt.reshape((3,1)) - (self.grid[:,0]-self.dpCenter[0])*ytilt.reshape((3,1))).T + self.dinfo.lastParam.tilt0
        self.simGrid = self.simGrid.reshape((-1, 3))

        if hasattr(self, 'padGrid'):
            self.padSimGrid = ((self.padGrid[:,1] - self.dpCenter[1])*xtilt.reshape((3,1)) - (self.padGrid[:,0]-self.dpCenter[0])*ytilt.reshape((3,1))).T + self.dinfo.lastParam.tilt0
            self.padSimGrid = self.padSimGrid.reshape((-1, 3))

        

    def setTilt0(self, tilt, refPoint):
        """Sets the central tilt of the simulation based on a reference point.

        Args:
            tilt (np.ndarray): The new tilt vector to be set as the center.
            refPoint (tuple): The (row, col) coordinates in the LARBED image
            that correspond to the new `tilt`.
        """
        # not sure if -1 is correct
        offset = self.inverse(refPoint[::-1]-1)[0,::-1] 
        ytilt, xtilt = pix2Tilt(self.dinfo.lastParam, self.initPixsiz)
        self.dinfo.lastParam.tilt0 = tilt - offset[1]*xtilt + offset[0]*ytilt
        

        if hasattr(self, 'simGrid'):
            self.updateSimGrid()


    
    # for debugging
    def getSampling(self):
        """Returns the raw sampling coordinates for debugging purposes."""
        grids = []
        sampleGrid = self.transformDP(self.grid*self.initPixsiz/self.pixsiz )
        # potentially can be optimized
        for idx in range(len(self.gInclude)):
            # calculate g vector and shift
            currGrid = sampleGrid + self.allshift + self.diskshift[idx]
            grids.append(currGrid)
        return grids


def pix2Tilt(param, pixsiz):
    """Converts a pixel displacement to a tilt vector in simulation units.

    Args:
        param (SimParams): The Bloch simulation parameters object, containing
        the reciprocal lattice vectors `gg` and `gh`.
        pixsiz (float): The size of a pixel in reciprocal space units (e.g., Å⁻¹).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the tilt vectors
        corresponding to a one-pixel displacement along the x and y axes.
    """
    xtilt = param.gg / scale(param.gmxr, param.gg) * pixsiz
    ytilt = param.gh / scale(param.gmxr, param.gh) * pixsiz
    return xtilt, ytilt


def split_array_by_lengths(A, B, axis=0):
    """Splits an array into subarrays of specified lengths.

    Args:
        A (np.ndarray): The array to be split.
        B (list[int]): A list of lengths for each subarray. The sum of lengths
        in B must equal the size of A along the specified axis.
        axis (int, optional): The axis along which to split the array.
        Defaults to 0.

    Returns:
        list[np.ndarray]: A list of subarrays.

    Raises:
        ValueError: If the sum of lengths in B does not match the array size.
    """
    # Verify that sum(B) equals len(A)
    if np.sum(B) != A.shape[axis]:
        raise ValueError("The sum of elements in B must equal the length of A.")
    return np.split(A, np.cumsum(B[:-1]), axis=axis)
    

def rotation(vec, theta):
    """Rotates a 2D vector by a given angle.

    Args:
        vec (np.ndarray): The 2D vector to rotate.
        theta (float): The rotation angle in radians.

    Returns:
        np.ndarray: The rotated 2D vector.
    """
    rot = np.array([[np.sin(theta), np.cos(theta)], [-np.cos(theta), np.sin(theta)]])
    return np.dot(rot, vec)
