from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial
from pathlib import Path
from itertools import combinations_with_replacement

import numpy as np

from pyextal.LucyRichardson import lucy_Richardson, DQE
from pyextal.symmetry import GroupSymmetry
from pyextal.callBloch import lookupSF, e2Xray
from pyextal.metric import scale
from pyextal.blochwave import spgr


@dataclass
class BaseDiffractionInfo:
    """Base class for storing diffraction information and refined results.

    Attributes:
        dp (np.ndarray): The diffraction pattern as a NumPy array of np.float32.
        thickness (float): Thickness of the sample in Angstroms.
        tiltX (float): Tilt angle around the x-axis in degrees.
        tiltY (float): Tilt angle around the y-axis in degrees.
        gl (float): X-axis length in experimental diffraction pattern space.
        datpath (str): Path to the .dat file containing crystal information.
    """
    dp: np.ndarray[np.float32]
    thickness: float
    tiltX: float
    tiltY: float
    gl: float
    datpath: str

    def __post_init__(self):
        """Initializes additional attributes after the dataclass is created."""
        self._includeBeam = None
        self.beamGroup = None
        self.beam2Group = None
        self.structureFactor = None
        self.beamDict = None
        self.lastParam = None

    
    # the last parameters (eigenvalue and vectors) used in the simulation
    # param: SimParams = field(default_factory=None)


    @property
    def includeBeam(self):
        """Gets the list of Miller indices for beams included in the simulation."""
        return self._includeBeam
    
    
    @includeBeam.setter
    def includeBeam(self, beams: np.ndarray):
        """Sets the beams to be included in the simulation and initializes related properties.

        This setter initializes the list of included beams, groups them by symmetry,
        and pre-calculates their initial structure factors.

        Args:
            beams (np.ndarray): A list of Miller indices (h, k, l) to be included.
        """
        print('include beam initialized')
        self._includeBeam = beams

        # initialize the beamDict
        ### TODO: extract method and move to symmetry
        self.beamDict = defaultdict(list)
        Ugh_table = beams.reshape(-1,1,3) - beams.reshape(1,-1,3)
        for i in range(1, Ugh_table.shape[0]):
            for j in range(0, i,1):
                self.beamDict[tuple(Ugh_table[i,j])].append((i,j))

        # group all included beams into symmetry equivalent groups
        self.symmetry = GroupSymmetry(beams)

        # initialize the structure factor
        self.structureFactor = np.ones((len(self.symmetry.beamGroup), 4), dtype=np.float32)

        # populate structure factor
        for i, beams in enumerate(self.symmetry.beamGroup):
            self.structureFactor[i] = lookupSF([beams[0],])

    def updateSF(self, beam:tuple, value: np.ndarray[np.float32]):
        """Updates the structure factor for a given beam and its symmetric equivalents.

        Args:
            beam (tuple): The Miller index (h, k, l) of the beam to update.
            value (np.ndarray): A 4-element NumPy array containing the new structure
                factor values.

        Raises:
            ValueError: If the provided `value` is not a 4-element NumPy array.
        """
        if value.shape != (4,):
            raise ValueError('The structure factor should be a 4 element np.array')
        self.structureFactor[self.symmetry.beam2Group[beam]] = value        

        
        

    def getSF(self, beam:tuple) -> np.ndarray:
        """Retrieves the structure factor for a given beam.

        If the beam is part of a symmetry group, the group's structure factor is
        returned. Otherwise, it looks up the Independent Atom Model (IAM) value.

        Args:
            beam (tuple): The Miller index (h, k, l) of the beam.

        Returns:
            np.ndarray: The structure factor of the beam.
        """
        key = self.symmetry.beam2Group.get(beam, False)
        if key:
            return self.structureFactor[key]
        else:
            return lookupSF([beam,])[0]
        

    def getAllSF(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieves all unique structure factors and their corresponding beams.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - An array of all unique structure factors.
                - An array of one representative beam (h, k, l) for each factor.
        """
        return self.structureFactor, np.array(self.symmetry.beamGroup)[:,0,:]
    
    def save(self, path: str|Path):
        """Saves the diffraction information to a file.

        Note: This method is not yet implemented.

        Args:
            path (str | Path): The path to the file where the data will be saved.
        """
        pass

    def save_hkl(self, path:str|Path, glen:float=5.0, minOutput:float=1e-3)->None:
        """Saves structure factors to a .hkl file for VESTA Fourier synthesis.

        Args:
            path (str | Path): Path to the output .hkl file.
            glen (float, optional): Maximum length of the g-vector to include,
                in inverse Angstroms. Defaults to 5.0.
            minOutput (float, optional): The minimum IAM structure factor magnitude
                to be included in the output file. Defaults to 1e-3.
        """
        hkls = []
        
        with open(path, 'w') as f:
            f.write("pyextal\nGenerated\n")
            f.write(f"{'h':<5} {'k':<5} {'l':<5}   {'|Fo|':<12} {'|Fc|':<12} {'Fc(real)':<12} {'Fc(imag)':<12} {'sigma(F)':<12}\n")
            hkllist = list(combinations_with_replacement(range(0, 10), 3))
            hkllist.sort(key=lambda x: scale(self.lastParam.gmxr, np.array(x)))
            print(f"HKL list length: {len(hkllist)}")
            for hkl in hkllist[1:]:            
                if hkl == (0,0,0):continue
                hkl = np.array(hkl)
                if scale(self.lastParam.gmxr, hkl) < glen:
                    hkl = tuple(hkl)
                    hkls.append(hkl)
                    getSF = partial(e2Xray, getESF=lambda x:self.getSF(x))
                    getIAM = partial(e2Xray, getESF=lambda x:lookupSF([x,], IAM=True)[0])
                    sf = getSF(hkl)
                    iam = getIAM(hkl)
                    if np.linalg.norm(iam) < minOutput:continue                    
                    f.write(f"{hkl[0]:<5} {hkl[1]:<5} {hkl[2]:<5}   {np.linalg.norm(iam):<12.5f} {np.linalg.norm(sf):<12.5f} {sf[0]:<12.5f} {sf[1]:<12.5f} {0.0:<12.5f}\n")


@dataclass
class CBEDDiffractionInfo(BaseDiffractionInfo):
    """Stores and processes Convergent Beam Electron Diffraction (CBED) data.

    This class extends `BaseDiffractionInfo` with attributes and methods specific
    to CBED experiments, including detector parameters and background correction.

    Attributes:
        dtpar (list[float]): Detector DQE parameters [varB, delta, A, g, m].
        mtf (np.ndarray): The Modulation Transfer Function as a NumPy array.
        background (float): Background level of the diffraction pattern.
        numIter (int, optional): Number of iterations for Lucy-Richardson
            deconvolution. Defaults to 25.
    """
    dtpar: list[float]
    mtf: np.ndarray[np.float32]
    background: float
    numIter: int = 25


    # def __post_init__(self, deconv=True):
    #     super().__post_init__()
    #     if deconv:
    #         self.dp = lucy_Richardson(self.dp, self.mtf, self.background, self.numIter, *self.dtpar)


@dataclass
class LARBEDDiffractionInfo(BaseDiffractionInfo):
    """Stores and processes Large Angle Rocking Beam Electron Diffraction (LARBED) data.

    This class extends `BaseDiffractionInfo` with attributes specific to LARBED
    experiments, such as g-vector indices and variance maps.

    Attributes:
        gindex (np.ndarray): An array of g-vector indices.
        varianceMaps (np.ndarray, optional): Variance maps associated with the
            diffraction pattern. If not provided, it defaults to a copy of the
            diffraction pattern.
    """
    # to be implemented
    # some specific processing for LARBED
    gindex: np.ndarray
    varianceMaps: np.ndarray[np.float32] = None
    


    def __post_init__(self):
        """Initializes variance maps if they are not provided."""
        super().__post_init__()
        if self.varianceMaps is None:
            self.varianceMaps = self.dp.copy()


