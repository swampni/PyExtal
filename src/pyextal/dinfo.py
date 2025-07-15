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
    '''
    class for storing diffraction information, store the refined result

    args:
            dp: diffraction pattern np.float32 np.array
            thickness: thickness of the sample in Angstrom
            tiltX: tilt angle around x axis in degree
            tiltY: tilt angle around y axis in degree
            gl: xaxis length in experiemnt dp space
            datpath: path to the .dat file         
            dtpar : detector DQE parameters; [varB,delta,A,g,m]       
    '''
    dp: np.ndarray[np.float32]
    thickness: float
    tiltX: float
    tiltY: float
    gl: float
    datpath: str

    def __post_init__(self):
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
        '''return the included beams in simulation'''
        return self._includeBeam
    
    
    @includeBeam.setter
    def includeBeam(self, beams):
        '''
        initialize the include beams into an ordered dictionary
        key: (h,k,l)
        value: structure factor adjust ratio of the beam np.array([|U|, phase(U), |UA|, phase(UA)])
        auto initialize the structure factor to 1

        args:
            beams: list of (h,k,l) included
        
        '''
        print('include beam initialized')
        self._includeBeam = beams

        # initialize the beamDict
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
        '''update the structure factor'''
        if value.shape != (4,):
            raise ValueError('The structure factor should be a 4 element np.array')
        self.structureFactor[self.symmetry.beam2Group[beam]] = value        

        
        

    def getSF(self, beam:tuple):
        '''return the structure factor'''
        key = self.symmetry.beam2Group.get(beam, False)
        if key:
            return self.structureFactor[key]
        else:
            return lookupSF([beam,])[0]
        

    def getAllSF(self):
        '''return all the structure factor and one of the symmetry equivalent beam'''
        return self.structureFactor, np.array(self.symmetry.beamGroup)[:,0,:]
    
    def save(self, path):
        '''save the diffraction information to a hdf5 file'''
        pass

    def save_hkl(self, path:str|Path, glen:float=5.0, minOutput:float=1e-3)->None:
        '''
        save the hkl information to a .hkl file for vesta fourier syntheisizer
        args:
            path: path to the .hkl file
            glen: maximum lenght of the output gl vector, in Angstrom^-1
        '''
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
    '''
    class for storing CBED diffraction information, store the refined result

    args:
            dp: diffraction pattern np.float32 np.array
            thickness: thickness of the sample in Angstrom
            tiltX: tilt angle around x axis in degree
            tiltY: tilt angle around y axis in degree
            gl: xaxis length in experiemnt dp space
            datpath: path to the .dat file
            dtpar: parameters generated bloch simulation (gain and those stuff)
            mtf: modulation transfer function
            background: background of the diffraction pattern
    '''
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
    '''
    class for storing LARBED diffraction information, store the refined result

    args:
            dp: diffraction pattern np.float32 np.array
            thickness: thickness of the sample in Angstrom
            tiltX: tilt angle around x axis in degree
            tiltY: tilt angle around y axis in degree
            gl: xaxis length in experiemnt dp space
            datpath: path to the .dat file
            gindex: list of g indices

    '''
    # to be implemented
    # some specific processing for LARBED
    gindex: float
    varianceMaps: np.ndarray[np.float32] = None
    


    def __post_init__(self):
        if self.varianceMaps is None:
            self.varianceMaps = self.dp.copy()


