from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import pyextal.blochwave as blochwave
import pyextal.cbedpy as cbedpy
from pyextal.metric import sumx, scale
from pyextal.symmetry import GroupSymmetry, appliedSymmetry

@dataclass
class Data:
    """Data class to store the data of each tilt calculated by the bloch engine.

    Attributes:
        ndiag (int): Number of diagonal elements.
        itilt (tuple): (3,) tilt vector.
        ix (int): x index of the tilt.
        iy (int): y index of the tilt.
        gamma (np.ndarray): Eigenvalues.
        eigenvector (np.ndarray): Eigenvectors.
        cinv (np.ndarray): 1st column of the inverse of the eigenvectors.
    """
    ndiag: int
    itilt: tuple
    ix: int
    iy: int
    gamma: np.ndarray
    eigenvector: np.ndarray
    cinv: np.ndarray

    @property
    def vr(self):
        """Real part of the eigenvalues."""
        return np.real(self.gamma)
    
    @property
    def vi(self):
        """Imaginary part of the eigenvalues."""
        return np.imag(self.gamma)
    
    @property
    def ccr(self):
        """Real part of the eigenvectors."""
        return np.real(self.eigenvector)
    
    @property
    def cci(self):
        """Imaginary part of the eigenvectors."""
        return np.imag(self.eigenvector)
    


@dataclass
class SimParams:
    """Data class to store the data and parameters from the simulation.

    Attributes:
        ntilt (int): Number of tilts.
        nout (int): Number of reflections.
        nx (int): Half width of sampling region.
        hklout (np.ndarray): (3, nout) np.int32 np.array of reflection indices.
        disks (float): Disk radius.
        alpha (float): Angle incidence.
        tilt0 (np.ndarray): (3,) np.float32 np.array of incident tilt.
        gg (np.ndarray): (3,) np.float32 np.array horizontal reciprocal vector of dp.
        gh (np.ndarray): (3,) np.float32 np.array vertical reciprocal vector of dp.
        gmx (np.ndarray): (3, 3) np.float32 np.array of gram tensor (gram matrix).
        gmxr (np.ndarray): (3, 3) np.float32 np.array of inverse of gram tensor (gram matrix).
        snorm (np.ndarray): (3,) np.float32 np.array of surface normal.
        bigK (float): Magnitude of refraction adjusted wave vector.
        tilts (list): List of Data class, each element is the data of each tilt.
    """
    ntilt: int
    nout: int
    nx: int
    hklout: np.ndarray
    disks: float
    alpha: float
    tilt0: np.ndarray
    gg: np.ndarray
    gh: np.ndarray
    gmx: np.ndarray
    gmxr: np.ndarray
    snorm: np.ndarray
    bigK: float
    tilts: list

    def __post_init__(self):
        self.tilt0 = self.tilt0[0, :]
        self.hklout = self.hklout[:, :self.nout]

    def __call__(self, i1, itilt):
        """Update blochwave parameters and pass data to cbedp common block.
        callback for bloch engine intensity calculation! DO NOT CALL THIS DIRECTLY!
        Args:
            i1 (int): 1-based index for the tilt.
            itilt (np.ndarray): Array to be updated with the x, y index of the tilt.

        Returns:
            np.ndarray: Updated itilt array.
        """
        # update the blochwave parameter upon calling
        # take account of 1-based indexing in fortran
        it = i1-1
        # pass the data to cbedp common block
        cbedpy.smatrx.vr[:self.tilts[it].ndiag] = self.tilts[it].vr.copy()
        cbedpy.smatrx.vi[:self.tilts[it].ndiag] = self.tilts[it].vi.copy()
        cbedpy.smatrx.ccr[:self.nout,
                          :self.tilts[it].ndiag] = self.tilts[it].ccr.copy()
        cbedpy.smatrx.cci[:self.nout,:self.tilts[it].ndiag] = self.tilts[it].cci.copy()
        cbedpy.smatrx.cinv[:self.tilts[it].ndiag] = self.tilts[it].cinv.copy()
        cbedpy.smatrx.ndiag = self.tilts[it].ndiag

        # return the x, y index of the tilt
        itilt[0] = self.tilts[it].ix
        itilt[1] = self.tilts[it].iy
        return itilt

    def larbedCall(self, i1):
        """Pass data to cbedp common block for LARBED calculation.
        Callback for bloch engine intensity calculation! DO NOT CALL THIS DIRECTLY!

        Args:
            i1 (int): 1-based index for the tilt.

        Returns:
            np.ndarray: The tilt vector.
        """
        it = i1-1
        # pass the data to cbedp common block
        cbedpy.smatrx.vr[:self.tilts[it].ndiag] = self.tilts[it].vr.copy()
        cbedpy.smatrx.vi[:self.tilts[it].ndiag] = self.tilts[it].vi.copy()
        cbedpy.smatrx.ccr[:self.nout,
                          :self.tilts[it].ndiag] = self.tilts[it].ccr.copy()
        cbedpy.smatrx.cci[:self.nout,:self.tilts[it].ndiag] = self.tilts[it].cci.copy()
        cbedpy.smatrx.cinv[:self.tilts[it].ndiag] = self.tilts[it].cinv.copy()
        cbedpy.smatrx.ndiag = self.tilts[it].ndiag

        # return the tilt vector
        itilt = self.tilts[it].itilt        
        return np.array(itilt)
        

    # def larbedInternal(self, i1):
    #     it = i1 - 1
    #     itilt = self.tilts[it].itilt
    #     blochwave.smatrx.ndiag = self.tilts[it].ndiag
    #     return itilt



    def simParam(self):
        """Return the simulation parameters as a tuple.

        Returns:
            tuple: A tuple containing the simulation parameters.
        """
        return self.ntilt, self.nx, self.hklout, self.gmxr, self.tilt0, self.gg, self.gh, self.snorm, self.disks, self.bigK, self.alpha

    def store(self, filename='simParam.h5'):
        """Store the simulation parameters to a file in HDF5 format.

        Args:
            filename (str, optional): The name of the file to save to. 
                Defaults to 'simParam.h5'.
        """
        import h5py
        #create a dataset with ntilt, nout, nx, disks, alpha as attributes
        with h5py.File(filename, 'w') as f:
            f.attrs['ntilt'] = self.ntilt
            f.attrs['nout'] = self.nout
            f.attrs['nx'] = self.nx
            f.attrs['disks'] = self.disks
            f.attrs['alpha'] = self.alpha
            f.attrs['bigK'] = self.bigK
            group = f.create_group('diffraction_parameters')
            group.create_dataset('hklout', data=self.hklout)
            group.create_dataset('gmxr', data=self.gmxr)
            group.create_dataset('tilt0', data=self.tilt0)
            group.create_dataset('gg', data=self.gg)
            group.create_dataset('gh', data=self.gh)
            group.create_dataset('snorm', data=self.snorm)
            for i, tilt in enumerate(self.tilts):
                group = f.create_group(f'tilt{i}')
                group.create_dataset('eigenvalue', data=tilt.gamma)                
                group.create_dataset('eigenvector', data=tilt.eigenvector)
                group.create_dataset('cinv', data=tilt.cinv)
                group.create_dataset('tilt', data=tilt.itilt)



class DataCollector:
    """Data collector class to retrieve data as a callback function."""
    tilts = []

    def __call__(self, ndiag, itilt, ix, iy, nout, ib=None):
        """Callback function to collect data from bloch engine diagonalization.

        Args:
            ndiag (int): Number of diagonal elements.
            itilt (tuple): The tilt vector.
            ix (int): The x index of the tilt.
            iy (int): The y index of the tilt.
            nout (int): The number of reflections.
            ib (np.ndarray, optional): Beam indices. Defaults to None.

        Returns:
            int: Always returns 0.
        """
        # take account of 1-based indexing in fortran
        # store the data to Data class
        if ib is None: # MPI
            tilt = Data(ndiag, tuple(itilt), ix, iy, blochwave.bloch_module.v[:ndiag].copy(),
                        blochwave.bloch_module.cc[:nout, :ndiag].copy(), blochwave.bloch_module.cinv[:ndiag].copy())
            self.tilts.append(tilt)           
            return 0
        else: # Serial # bug here
            tilt = Data(ndiag, itilt, ix, iy, blochwave.bloch_module.v[:ndiag].copy(),
                    blochwave.bloch_module.cc[ib[:nout]-1, :ndiag].copy(), blochwave.bloch_module.cinv[:ndiag].copy())
            self.tilts.append(tilt)
            return 0


def updateUgh(reflections, values, beamDict=None):
    """Update the Ugh matrix in the common block with the given values.

    Args:
        reflections (list): List of tuples (nref,) for reflection indices,
            e.g., [(1, 1, 1), (1, 1, 0)].
        values (np.ndarray): (nref, 4) or (nref, 2) array of structure factor
            values. For (nref, 4): Ugh, phase, U'gh, phase. Ugh should always
            be positive. For (nref, 2): Ugh, U'gh.
        beamDict (dict, optional): Dictionary mapping beam pairs to their
            positions in the Ugh matrix. Defaults to None.
    """    
    if beamDict is None:
        ### TODO: extract method and move to symmetry
        beamDict = defaultdict(list)
        beams = blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T
        Ugh_table = beams.reshape(-1,1,3) - beams.reshape(1,-1,3)
        for i in range(1, Ugh_table.shape[0]):
            for j in range(0, i,1):
                beamDict[tuple(Ugh_table[i,j])].append((i,j))
    # get the plus and minus sign for centrosymmetric crystals
    phase = np.cos(np.deg2rad(lookupSF(reflections)[:,[1,3]]))

    for i, reflection in enumerate(reflections):        
        if values.shape[1] == 4:
            xfmag = values[i, 0]
            xfphase = np.deg2rad(values[i, 1])
            afmag = values[i, 2]
            afphase = np.deg2rad(values[i, 3])
            xfre, xfim = xfmag*np.cos(xfphase), xfmag*np.sin(xfphase)
            afre, afim = afmag*np.cos(afphase), afmag*np.sin(afphase)
        elif values.shape[1] == 2:
            
            xfre = values[i, 0]*phase[i, 0]
            afre = values[i, 1]*phase[i, 1]
            xfim, afim = 0, 0
    
        else:
            raise ValueError('values must be (nref, 4) or (nref, 2) np.float32 np.array')

        for ugh in beamDict[tuple(reflection)]:
            # print(ugh[0],ugh[1],)
            
            # print('-'*20)
            blochwave.bloch_module.ughr[ugh[0], ugh[1]] = xfre - afim
            blochwave.bloch_module.ughi[ugh[0], ugh[1]] = xfim + afre
            blochwave.bloch_module.ughr[ugh[1], ugh[0]] = xfre + afim
            blochwave.bloch_module.ughi[ugh[1], ugh[0]] = -xfim + afre
        
        for ugh in beamDict[tuple(-np.array(reflection))]:
            # print(ugh[0],ugh[1],)
            

            # print('-'*20)
            blochwave.bloch_module.ughr[ugh[0], ugh[1]] = xfre + afim
            blochwave.bloch_module.ughi[ugh[0], ugh[1]] = -xfim + afre
            blochwave.bloch_module.ughr[ugh[1], ugh[0]] = xfre - afim
            blochwave.bloch_module.ughi[ugh[1], ugh[0]] = xfim + afre

def updateSymUgh(reflections, values, sym=None):
    """Update the Ugh matrix with given values, including all symmetry-related beams.

    Args:
        reflections (list): List of tuples (nref,) for reflection indices,
            e.g., [(1, 1, 1), (1, 1, 0)].
        values (np.ndarray): (nref, 4) or (nref, 2) array of structure factor
            values. For (nref, 4): Ugh, phase, U'gh, phase. For (nref, 2): Ugh, U'gh.
        sym (GroupSymmetry, optional): GroupSymmetry class instance for symmetry-related
            beams. Defaults to None.
    """
    if sym is None:
        beams = blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T
        sym = GroupSymmetry(beams)    
    
    
    
    
    for ith, reflection in enumerate(reflections):
        phase = np.cos(np.deg2rad(lookupSF(reflections)[:,[1,3]]))        
        if sym.centro:
            xfre = values[ith, 0]*phase[ith, 0]
            afre = values[ith, 1]*phase[ith, 1]
            for i,j in sym.getPos(tuple(reflection)):
                # print(i,j)
                # print(i,blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[i])
                # print(j,blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[j])
                # print(blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[i] - blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[j])
                # print('-'*20)
                blochwave.bloch_module.ughr[i, j] = xfre 
                blochwave.bloch_module.ughi[i, j] = afre
        else:
            xfmag = values[ith, 0]
            xfphase = np.deg2rad(values[ith, 1])
            afmag = values[ith, 2]
            afphase = np.deg2rad(values[ith, 3])
            xfre, xfim = xfmag*np.cos(xfphase), xfmag*np.sin(xfphase)
            afre, afim = afmag*np.cos(afphase), afmag*np.sin(afphase)
            for i,j in sym.getPos(tuple(reflection)):
                # print(i,blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[i])
                # print(j,blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[j])
                # print('-'*20)
                # xfre, xfim = np.real(xf), np.imag(xf)
                # afre, afim = np.real(af), np.imag(af)
                blochwave.bloch_module.ughr[i, j] = xfre - afim
                blochwave.bloch_module.ughi[i, j] = xfim + afre
                # blochwave.bloch_module.ughr[j, i] = xfre + afim
                # blochwave.bloch_module.ughi[j, i] = -xfim + afre
            # print('*'*20)
            for i,j in sym.getPos(tuple(-np.array(reflection))):
                # xfre, xfim = np.real(xf), np.imag(xf)
                # afre, afim = np.real(phase), np.imag(phase)
                # print(i,blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[i])
                # print(j,blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T[j])
                # print('-'*20)
                blochwave.bloch_module.ughr[i, j] = xfre + afim
                blochwave.bloch_module.ughi[i, j] = -xfim + afre
                # blochwave.bloch_module.ughr[j, i] = xfre - afim
                # blochwave.bloch_module.ughi[j, i] = xfim + afre

        
        

def symmetrizeVHKL():
    """Symmetrize the structure factor adjusted in the .dat file."""
    hkl = blochwave.difpar.hklf[:3,:blochwave.difpar.nref].T
    hkl = tuple(tuple(x) for x in hkl)
    hkls = []
    values = []
    for i, reflection in enumerate(hkl):
        beam, sf = appliedSymmetry(reflection, blochwave.difpar.vhkl[:4,i])
        hkls.extend(beam)
        values.append(sf)
    values = np.vstack(values).T
    blochwave.difpar.nref = len(hkls)
    blochwave.difpar.hklf[:3,:blochwave.difpar.nref] = np.array(hkls).T    
    blochwave.difpar.vhkl[:4,:blochwave.difpar.nref] = values


        

# def adjustSF(reflections, values):
#     """Adjust the structure factor to the given values.

#     Args:
#         reflections (list): List of tuples (nref,) for reflection indices,
#             e.g., [(1, 1, 1), (1, 1, 0)].
#         values (np.ndarray): (4, nref) array of structure factor values
#             (Ugh, phase, U'gh, phase).
#     """
#     nref = len(reflections)
#     hklf = np.zeros((3, nref), dtype=np.int32, order='F')
#     for i, reflection in enumerate(reflections):
#         hklf[:, i] = np.array([index for index in reflection])
#     if values.shape[0] != 4 or values.shape[1] != nref:
#         raise ValueError('values must be (4, nref) np.float32 np.array')
#     if np.isfortran(values) is False:
#         values = np.asfortranarray(values)
#     blochwave.difpar.nref = nref
#     blochwave.difpar.hklf[:3, :nref] = hklf
#     blochwave.difpar.vhkl[:4, :nref] = values

def lookupSF(reflections, IAM=False):
    """Look up the default structure factor for given reflections.
    The default values are from Bird and King, Acta Cryst. A 46 (1990) 202.

    Args:
        reflections (list): List of tuples (nref,) for reflection indices,
            e.g., [(1, 1, 1), (1, 1, 0)].
        IAM (bool, optional): If True, use the independent atom model (IAM) for
            all structure factors. If False, return the adjusted structure
            factor from the .dat file. Defaults to False.

    Returns:
        np.ndarray: (nref, 4) array of structure factor values
        (Ugh, phase (deg), U'gh, phase (deg)).
    """
    values = np.zeros((len(reflections), 4), dtype=np.float32)
    if IAM:
        for i, reflection in enumerate(reflections):
            _,xfre, xfim, afre, afim = blochwave.defaultsf(3, reflection)
            values[i,0] = np.abs(xfre+1j*xfim)
            values[i,1] = np.angle(xfre+1j*xfim, deg=True)
            values[i,2] = np.abs(afre+1j*afim)
            values[i,3] = np.angle(afre+1j*afim, deg=True)

        return values
    else:
        # didn't take account the symmetry of the crystal
        adjusted = blochwave.difpar.hklf[:3,:blochwave.difpar.nref].T
        adjusted = tuple(tuple(x) for x in adjusted)
        for i, reflection in enumerate(reflections):
            if tuple(reflection) not in adjusted:
                _,xfre, xfim, afre, afim = blochwave.defaultsf(3, reflection)
                values[i,0] = np.abs(xfre+1j*xfim)
                values[i,1] = np.angle(xfre+1j*xfim, deg=True)
                values[i,2] = np.abs(afre+1j*afim)
                values[i,3] = np.angle(afre+1j*afim, deg=True)
            else:
                # get the adjusted structure factor from the .dat file
                values[i, :] = blochwave.difpar.vhkl[:4, adjusted.index(tuple(reflection))].copy()
        
        return values
        

def lookupReflections():
    """Look up all non-zero reflections.

    Returns:
        list: A list of all reflections excluding the (0, 0, 0) beam.
    """
    reflections = blochwave.difpar.hkl[:,1:].T
    return [x for x in reflections if  np.any(x != [0, 0, 0])]


def bloch(fname, t_x=0, t_y=0, reflections = None, values=None, HKL=False, subaper=0, subnx = None, subny = None, pixsiz=0, ncores=1, xaxis=None, dryrun=False):
    """Simulate the diffraction pattern point-to-point (no interpolation).

    Args:
        fname (str): Path to the .dat file.
        t_x (float, optional): Tilt in the x direction (deg). Defaults to 0.
        t_y (float, optional): Tilt in the y direction (deg). Defaults to 0.
        reflections (list, optional): List of tuples (nref,) for reflection to adjust
            indices, e.g., [(1, 1, 1), (1, 1, 0)]. Defaults to None.
        values (np.ndarray, optional): (4, nref) array of structure factor
            values (Ugh, phase (deg), U'gh, phase (deg)) to adjust. Defaults to None.
        HKL (bool, optional): If True, return the hkl indices of the included
            beams. Defaults to False.
        subaper (int, optional): Aperture function.
            0 or None: No aperture.
            1: Circular aperture of input size nx (Standard CBED).
            2: Custom circular aperture of radius (subnx1+subnx2).
            3: Custom rectangular aperture of shape (subnx),(subny).
            Defaults to 0.
        subnx (int, optional): Dependent on subaper parameter. Defaults to None.
        subny (int, optional): Dependent on subaper parameter. Defaults to None.
        pixsiz (float, optional): Pixel size in diffraction space. Defaults to 0.
        ncores (int, optional): Number of cores to use. Defaults to 1.
        xaxis (np.ndarray, optional): (3,) array for the x-axis of the
            diffraction space. Defaults to None.
        dryrun (bool, optional): If True, perform a dry run to estimate memory
            usage without running the full simulation. Defaults to False.

    Returns:
        SimParams or float: SimParams object containing simulation results, or
        estimated memory usage if dryrun is True.
        np.ndarray, optional: If HKL is True, returns a (nbeams, 3) array of
        reflection indices.
    """
    
    #convert fname to bytes array
    fname = np.array(bytes(fname, encoding='utf-8'))  

    # parse the .dat file
    header = blochwave.bloch_parse(fname, t_x, t_y)
    if header[0] == -1:
        raise ValueError('error parsing the .dat file')
    nx = header[1]
    disks = header[3]
    gg = header[6]
    # set adj = -1 read structure factor from .dat
    # set adj = other value, do not read structure factor from .dat, adjust by calling adjustSF
    # phase is in degree

    if reflections is None or values is None:
        # blochwave.difpar.nref = 0 
        pass
    else:   
        updateUgh(reflections, values)
    if subaper == 0 or subaper == 1:
        subnx = np.zeros(2, dtype=np.int32)
        subny = np.zeros(2, dtype=np.int32)
        subnx[0] = nx
    elif subaper == 2:
        if sum(subnx) > nx:
            raise ValueError('aperture bigger than set value in .dat file')
    elif subaper == 3:
        if np.max(np.abs(subnx)) > nx or np.max(np.abs(subny)) > nx:
            raise ValueError('aperture bigger than set value in .dat file')
    else:
        raise ValueError('subaper must be 0, 1, 2, or 3') 

    if xaxis is None:
        xaxis = header[6]
    else:
        if len(xaxis) != 3:
            raise ValueError('xaxis must have 3 elements')
        xaxis = np.array(xaxis, dtype=np.float32, order='F')

    if pixsiz==0:
        pixsiz = disks*scale(blochwave.gram.gmxr, gg)/nx

    # initialize data collector
    f = DataCollector()
    if dryrun:
        memoryUse = blochwave.dryrun(header[0], header[5][0,:], gg, xaxis, pixsiz, subaper, subnx, subny, ncores)
        return memoryUse
    # start simulation
    ntilt = blochwave.bloch_run(header[0], header[5][0,:], gg, xaxis, pixsiz, subaper, subnx, subny, ncores, collector=f)
    if ntilt == -1:
        raise ValueError('error running the simulation')
    # save the simulation results
    param = SimParams(ntilt, *header, blochwave.gram.gmx.copy(), blochwave.gram.gmxr.copy(),
                      blochwave.difpar.snorm.copy(), blochwave.difpar.bigk.copy(), f.tilts)
    # clear the data collector
    DataCollector.tilts = []
    
    if HKL:
        return param, blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T
    return param

def bloch_parse(fname, t_x=0, t_y=0, HKL=False):
    """Parse the .dat file.

    Args:
        fname (str): Path to the .dat file.
        t_x (float, optional): Tilt in the x direction (deg). Defaults to 0.
        t_y (float, optional): Tilt in the y direction (deg). Defaults to 0.
        HKL (bool, optional): If True, return the hkl indices of the included
        beams. Defaults to False.

    Returns:
        SimParams: An empty SimParams object with simulation info.
        np.ndarray, optional: If HKL is True, returns a (nbeams, 3) array of
        reflection indices.
    """
    #convert fname to bytes array
    fname = np.array(bytes(fname, encoding='utf-8'))  
    # parse the .dat file
    header = blochwave.bloch_parse(fname, t_x, t_y)
    if header[0] == -1:
        raise ValueError('error parsing the .dat file')
    param = SimParams(0, *header, blochwave.gram.gmx.copy(), blochwave.gram.gmxr.copy(),
                      blochwave.difpar.snorm.copy(), blochwave.difpar.bigk.copy(), [])
    if HKL:
        return param, blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T
    return param

def tilt_run(param, tilts, indices=None, ncores=1, HKL=False):
    """Run the simulation for a given set of tilts. The eigenvalue/vectors are stored in blochwave.bloch_module

    Args:
        param (SimParams): SimParams object with simulation parameters.
        tilts (np.ndarray): (n, 3) array of tilt vectors.
        indices (np.ndarray, optional): (n,) array of indices. Defaults to None.
        ncores (int, optional): Number of cores to use. Defaults to 1.
        HKL (bool, optional): If True, return the hkl indices of the included
        beams. Defaults to False.

    Returns:
        SimParams: The updated SimParams object.
        np.ndarray, optional: If HKL is True, returns a (nbeams, 3) array of
        reflection indices.
    """
    # simulate by the output setting in .dat file
    if indices is None:
        indices = np.arange(param.hklout.shape[1])
        
    # reallocate the tilt array if it is already initialized
    if blochwave.eigen.tilt is None:
        blochwave.eigen.allocate_tilt(tilts.shape[0])       
    elif blochwave.eigen.tilt.shape != tilts.T.shape:        
        blochwave.eigen.deallocate_arrays_eigen()
        blochwave.eigen.allocate_tilt(tilts.shape[0])       
    
    blochwave.eigen.tilt = tilts.T
    # take account of 1-based indexing in fortran
    blochwave.tilt_run(param.nout, indices+1, ncores)
    
    param.ntilt = tilts.shape[0]
    param.tilts = []
    if HKL:
        return param, blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T
    return param




def bloch_run(param, HKL=False, subaper=0, subnx = None, subny = None, pixsiz=0, ncores=1, xaxis=None):
    """Construct a tilt net and run the simulation.

    Args:
        param (SimParams): SimParams object with simulation parameters.
        HKL (bool, optional): If True, return the hkl indices of the included
            beams. Defaults to False.
        subaper (int, optional): Aperture function.
            0 or None: No aperture.
            1: Circular aperture of input size nx (Standard CBED).
            2: Custom circular aperture of radius (subnx1+subnx2).
            3: Custom rectangular aperture of shape (subnx),(subny).
            Defaults to 0.
        subnx (int, optional): Dependent on subaper parameter. Defaults to None.
        subny (int, optional): Dependent on subaper parameter. Defaults to None.
        pixsiz (float, optional): Pixel size in diffraction space. Defaults to 0.
        ncores (int, optional): Number of cores to use. Defaults to 1.
        xaxis (np.ndarray, optional): (3,) array for the x-axis of the
        diffraction space. Defaults to None.

    Returns:
        SimParams: The updated SimParams object.
        np.ndarray, optional: If HKL is True, returns a (nbeams, 3) array of
        reflection indices.
    """
    
    if pixsiz==0:
        pixsiz = param.disks*scale(blochwave.gram.gmxr, param.gg)/param.nx
    else:
        param.nx = int(np.floor(param.disks*scale(blochwave.gram.gmxr, param.gg)/pixsiz))

    if subaper == 0 or subaper == 1:
        subnx = np.zeros(2, dtype=np.int32)
        subny = np.zeros(2, dtype=np.int32)
        subnx[0] = param.nx
    

    if xaxis is None:
        xaxis = param.gg
    else:
        if len(xaxis) != 3:
            raise ValueError('xaxis must have 3 elements')
        xaxis = np.array(xaxis, dtype=np.float32, order='F')

    

    # initialize data collector
    f = DataCollector()
    
    # set t_x, t_y for tilt
    # print(subaper, subnx1,subnx2,subny1, subny2)
    # start simulation
    ntilt = blochwave.bloch_run(param.nout, param.tilt0, param.gg, xaxis, pixsiz, subaper, subnx, subny, ncores, collector=f)
    if ntilt == -1:
        raise ValueError('error running the simulation')
    # save the simulation results
    
    param.ntilt = ntilt
    param.tilts = f.tilts
    
    # clear the data collector
    DataCollector.tilts = []

    if HKL:
        return param, blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T
    return param

def simulate(fname, thickness, displayParam, t_x=0, t_y=0, reflections = None, values=None, HKL=False, subaper=0, subnx = None, subny = None, pixsiz=0, ncores=1, dryrun=False):
    """Wrapper for bloch_run and cbedp to simulate and return the diffraction pattern.

    Args:
        fname (str): Path to the .dat file.
        thickness (float): Thickness of the sample in Angstroms.
        displayParam (list): List of parameters for display:
            [x0, y0, gl, xs, ys, theta].
        t_x (float, optional): Tilt in the x direction (deg). Defaults to 0.
        t_y (float, optional): Tilt in the y direction (deg). Defaults to 0.
        reflections (list, optional): List of tuples (nref,) for reflection
            indices, e.g., [(1, 1, 1), (1, 1, 0)]. Defaults to None.
        values (np.ndarray, optional): (4, nref) array of structure factor
            values (Ugh, phase (deg), U'gh, phase (deg)). Defaults to None.
        HKL (bool, optional): If True, return the hkl indices of the included
            beams. Defaults to False.
        subaper (int, optional): Aperture function.
            0 or None: No aperture.
            1: Circular aperture of input size nx (Standard CBED).
            2: Custom circular aperture of radius (subnx1+subnx2).
            3: Custom rectangular aperture of shape (subnx),(subny).
            Defaults to 0.
        subnx (int, optional): Dependent on subaper parameter. Defaults to None.
        subny (int, optional): Dependent on subaper parameter. Defaults to None.
        pixsiz (float, optional): Pixel size in diffraction space. Defaults to 0.
        ncores (int, optional): Number of cores to use. Defaults to 1.
        dryrun (bool, optional): If True, perform a dry run to estimate memory
        usage. Defaults to False.

    Returns:
        np.ndarray: (xs, ys) array of the diffraction pattern.
        SimParams: SimParams object with simulation parameters.
        np.ndarray, optional: If HKL is True, returns a (nout, 3) array of
        reflection indices.
    """

    param = bloch(fname, t_x, t_y, reflections, values, HKL=False, subaper=subaper, subnx=subnx, subny=subny, pixsiz=pixsiz, ncores=ncores, dryrun=dryrun)
    if dryrun:
        print(f'estimate memory usage: {param} bytes')
        return
    x0, y0, gl, xs, ys, theta = displayParam
    # calculate the diffraction pattern
    dp = cbedpy.cbedp(thickness, x0, y0, gl, 2048, theta, *param.simParam(), param)

    # cbedp always generate a 2048x2048 image, so we need to crop it to the desired size
    # easier for generating non-square images
    if HKL:
        xs = int(xs)
        ys = int(ys)
        return dp.reshape(2048,2048)[:xs, :ys], param, blochwave.difpar.hkl[:,:blochwave.difpar.nbeams].T
    else:
        return dp.reshape(2048,2048)[:xs, :ys], param
    

def LARBED(param, thickness, height=None, width=None, tiltmap=False):
    """Calculate the LARBED pattern for a given thickness.

    Args:
        param (SimParams): SimParams object with simulation parameters.
        thickness (int or np.ndarray): Thickness of the sample in Angstroms.
            Can be a single int or a (ntilt,) array.
        height (int, optional): Height of the diffraction pattern. Defaults to None.
        width (int, optional): Width of the diffraction pattern. Defaults to None.
        tiltmap (bool, optional): If True, return the tilt map. Defaults to False.

    Returns:
        np.ndarray: (nout, height, width) array of diffraction patterns.
        np.ndarray, optional: If tiltmap is True, returns a (nx, nx, 3)
        array of the tilt map.
    """    
    
    if isinstance(thickness, np.ndarray):
        thickness = thickness.flatten()
        if thickness.shape[0] != param.ntilt:
            raise ValueError('thickness must have the same length as ntilt')
        dp = cbedpy.larbed2(thickness, *param.simParam()[1:], param.larbedCall)
    elif isinstance(thickness, (int, float, np.number)):
        dp = cbedpy.larbed(thickness,*param.simParam(), param.larbedCall)
    else:
        raise ValueError(f'thickness must be int or np.ndarray, not {type(thickness)}')
    # dp = blochwave.larbed(thickness, *param.simParam(), param.larbedInternal)
    # dp = dp[:param.nout*param.ntilt]

    if height is None or width is None:
        height = width = param.nx*2 + 1
    dp = dp.reshape(param.nout, height, width)
    dp = np.flip(dp, axis=1)

    if not tiltmap:return dp

    tiltx = param.disks*param.gg/param.nx
    tilty = param.disks*param.gh/param.nx

    tiltmap = np.zeros(shape=(height, width, 3), dtype=np.float32, order='F')
    for data in param.tilts:
        tilt = data.ix*tiltx + data.iy*tilty + param.tilt0
        tiltmap[data.iy+param.nx, data.ix+param.nx] = tilt
    return dp, tiltmap

def LARBED_tilt(param, thickness, nout):
    """Calculate the LARBED pattern for a given thickness with eigenvector/values store in blochwave.bloch_module.

    Args:
        param (SimParams): SimParams object with simulation parameters.
        thickness (int or np.ndarray): Thickness of the sample in Angstroms.
            Can be a single int or a (ntilt,) array.
        nout (int): Number of output points.

    Returns:
        np.ndarray: (nout, ntilt) array of diffraction patterns.
    """
    # single thickness value
    
    # assign thickness for each tilt
    if isinstance(thickness, np.ndarray):
        thickness = thickness.flatten()
        if thickness.shape[0] != param.ntilt:
            raise ValueError('thickness must have the same length as ntilt')
        return blochwave.larbed2(thickness, param.alpha, param.gg, nout).reshape(nout, param.ntilt)
    elif isinstance(thickness, (int, float, np.number)):
        return blochwave.larbed(thickness, param.ntilt, param.alpha, param.gg, nout).reshape(nout, param.ntilt)
    else:
        raise ValueError('thickness must be int or np.ndarray')
    

def terminate():
    """Deallocate the arrays in the blochwave module."""
    blochwave.eigen.deallocate_arrays_eigen()
    blochwave.difpar.deallocate_arrays_difpar()
    blochwave.bloch_module.deallocate_arrays_bloch()
    blochwave.mpiinfo.mpiinfo_despawn()
    
    

def calibrateLARBED(param, gl):
    """Calibrate LARBED parameters.

    Args:
        param (SimParams): SimParams object with simulation parameters.
        gl (float): Geometric scaling factor.

    Returns:
        tuple: A tuple containing the side length and scale factor.
    """
    conv = param.disks
    sampling = param.nx
    side = np.sqrt(sumx(param.gmxr, param.gg, param.gg))/param.bigK/np.pi*180*conv*2
    scale_factor = gl*conv / sampling
    return side, scale_factor


def calibrateCBED(dp, center, centerTiltx, centerTilty, gl, param):
    """Calibrate CBED parameters.

    Args:
        dp (np.ndarray): The diffraction pattern.
        center (tuple): The center of the diffraction pattern.
        centerTiltx (float): The tilt in the x direction at the center.
        centerTilty (float): The tilt in the y direction at the center.
        gl (float): Geometric scaling factor.
        param (SimParams): SimParams object with simulation parameters.

    Returns:
        tuple: A tuple containing the x and y tilt maps.
    """
    gDegree = np.sqrt(sumx(param.gmxr, param.gg, param.gg))/param.bigK/np.pi*180
    gUnit = gDegree/gl
    ytilt = np.arange(0, (dp.shape[0])*gUnit, gUnit)
    xtilt = np.arange(0, (dp.shape[1])*gUnit, gUnit)
    xTiltMap, yTiltMap = np.meshgrid(xtilt, ytilt)
    xTiltMap -= xTiltMap[center[0],center[1]]
    yTiltMap -= yTiltMap[center[0],center[1]]
    xTiltMap += centerTiltx
    yTiltMap += centerTilty
    return xTiltMap, yTiltMap  

def wavelength():
    """Return the wavelength of the electron beam in Angstroms.

    Returns:
        float: The wavelength in Angstroms.
    """
    kv = blochwave.difpar.kv
    return 0.3878314/np.sqrt(kv*(1.0+0.97846707E-03*kv))

def tilt(param, t_x, t_y):
    """Calculate the tilt vector for a given tilt in x and y directions.

    Args:
        param (SimParams): SimParams object with simulation parameters.
        t_x (float): Tilt in the x direction (deg).
        t_y (float): Tilt in the y direction (deg).
    """
    
    
    gl = scale(param.gmxr, param.gg)
    wavel = wavelength()
    param.tilt0 = param.tilt0 - np.deg2rad(t_x)*param.gg/gl/wavel \
                              - np.deg2rad(t_y)*param.gh/gl/wavel
    
def tiltUnit(param):
    """Calculate the tilt unit vectors.

    Args:
        param (SimParams): SimParams object with simulation parameters.

    Returns:
        tuple: A tuple containing the tilt unit vectors in the x and y directions.
    """
    return param.disks/param.nx*param.gg, param.disks/param.nx*param.gh

def e2Xray(beam:tuple, getESF:callable) -> np.ndarray:
    """Calculate the X-ray scattering factor for a given beam.

    Args:
        beam (tuple): The beam for which the X-ray scattering factor is to be
            calculated.
        getESF (callable): A function to get the electron scattering factor.

    Returns:
        np.ndarray: The X-ray scattering factor for the given beam.
    """
    # xf = dinfo.getSF(beam)    
    xf = getESF(beam)
    # Get the structure factor for the given beam
    phase1 = np.deg2rad(xf[1])
    phase2 = np.deg2rad(xf[3])
    xf = np.array([xf[0]*np.cos(phase1), xf[0]*np.sin(phase1), xf[2]*np.cos(phase2), xf[2]*np.sin(phase2)])
    return blochwave.sfmanager.e2xray(beam, xf)




