"""Symmetry Operations and Grouping.

This module provides tools for handling crystallographic symmetry operations.
It includes a class for grouping symmetry-equivalent reflections and functions
for applying symmetry operations to find all equivalent reflections for a given
vector. The functionalities rely on the `spgr` common block from the Fortran
Bloch wave code, which must be initialized before use.
"""
from collections import OrderedDict, defaultdict

import numpy as np 

from pyextal.blochwave import spgr


class GroupSymmetry:
    """Groups symmetry-equivalent beams and manages their relationships.

    This class takes a list of beams (reflections) and groups them based on the
    crystal's symmetry operations. It creates a mapping from each beam to its
    symmetry group and stores the positions of these beams in the `Ugh` matrix
    used in Bloch wave calculations.

    Attributes:
        centro (bool): True if the crystal structure is centrosymmetric.
        beamGroup (list): A list of lists, where each inner list contains a group
            of symmetry-equivalent beams.
        beam2Group (OrderedDict): A mapping from each beam (tuple) to its group
            index in `beamGroup`.
        Ugh_dict (defaultdict): A dictionary mapping a group index to a list of
            (i, j) positions in the `Ugh` matrix.
        gp (np.ndarray): The symmetry operations from the `spgr` module.
        phaseDict (dict): A dictionary to store phase factors for each beam.
    """
    def __init__(self, beams):
        """Initializes the GroupSymmetry object and performs the grouping.

        Args:
            beams (np.ndarray): A NumPy array of shape `(n_beams, 3)` containing
                the Miller indices (h, k, l) of the beams to be grouped.
        """
        self.centro = bool(spgr.centro)
        self.beamGroup = [] # store the group of beams that are symmetry equivalent
        self.beam2Group = OrderedDict() # each beam to its group
        self.Ugh_dict = defaultdict(list) # store the symmetry equivalent beams positions in Ugh matrix
        self.beamDict = defaultdict(list) # store the beam occurrence in Ugh matrix
        self.gp = getSymmetry() # symmetry operations
        self.phaseDict = dict() # store the phase factor for each beam

        record = set() # track the beams that have been assigned to a group
        
        # collect the indices for same Ugh
        Ugh_table = beams.reshape(-1,1,3) - beams.reshape(1,-1,3)        
        for i in range(1, Ugh_table.shape[0]):
            for j in range(0,i,1):
                beam  = tuple(Ugh_table[i,j])
                # check if the beam is already assigned to a group
                if beam not in record:                    
                    # not included in any group, add a new group
                    allSym = appliedSymmetry(beam)
                    self.beamGroup.append(allSym)
                    igroup = len(self.beamGroup) - 1
                    # assign all symmetry equivalent beams to the same group
                    for symBeam in allSym:
                        self.beam2Group[symBeam] = igroup
                        record.add(symBeam)
                    if not self.centro:
                        invBeam = tuple(-Ugh_table[i,j])
                        # not included in any group, add a new group
                        allSym = appliedSymmetry(invBeam)
                        self.beamGroup.append(allSym)
                        igroup = len(self.beamGroup) - 1
                        # assign all symmetry equivalent beams to the same group
                        for symBeam in allSym:
                            self.beam2Group[symBeam] = igroup
                            record.add(symBeam)

                # currentGroup = self.getGroup(beam)
                # index = currentGroup.index(beam)

                # calculate the phase factor for symmetry related beams, 
                # see Waser et al. (1955), Acta Cryst. DOI:10.1107/S0365110X55001862
                # phase = np.exp(-2*np.pi*1j*np.dot(self.gp[index, 9:], currentGroup[0]))
                # self.phaseDict[beam] = phase
                self.Ugh_dict[self.beam2Group[beam]].append((i,j))

                if self.centro:
                    self.Ugh_dict[self.beam2Group[beam]].append((j,i))
                else:
                    igroup = self.beam2Group[beam]
                    if igroup % 2 == 0 : igroup += 1
                    else: igroup -= 1                    
                    self.Ugh_dict[igroup].append((j,i))
                
                # # do the same for the inverse beam
                # currentGroup = self.getGroup(invBeam)
                # index = currentGroup.index(invBeam)
                # phase = np.exp(-2*np.pi*1j*np.dot(self.gp[index, 9:], currentGroup[0]))
                # self.phaseDict[invBeam] = phase
                
                # self.Ugh_dict[self.beam2Group[invBeam]].append((i,j,phase))
        print('group symmetry initialized', flush=True)
    
    def getGroup(self, beam):
        """Retrieves the symmetry group for a given beam.

        Args:
            beam (tuple): The Miller index (h, k, l) of the beam.

        Returns:
            list: A list of all beams that are symmetry-equivalent to the input beam.
        """
        return self.beamGroup[self.beam2Group[beam]]
    
    def getPos(self, beam):
        """Yields the Ugh matrix positions for a given beam's symmetry group.

        Args:
            beam (tuple): The Miller index (h, k, l) of the beam.

        Yields:
            tuple[int, int]: The (row, column) indices in the Ugh matrix for each
            member of the beam's symmetry group.
        """        
        beam = tuple(beam)
        # currentPhase = self.phaseDict[beam]
        for i,j in self.Ugh_dict[self.beam2Group[beam]]:
            yield i,j#,phase/currentPhase


def appliedSymmetry(gg, sf=None):
    """Calculates symmetry-equivalent reflections and their structure factors.

    This function applies the crystal's symmetry operations to a given reflection
    vector `gg` to find all unique equivalent reflections. If structure factors
    `sf` are provided, it also calculates the corresponding structure factors for
    each equivalent reflection, including phase shifts.

    Note:
        The `spgr` module from the Fortran code must be initialized
        (e.g., by running a simulation) before calling this function.

    Args:
        gg (array_like): A 1D sequence (list, tuple, or np.ndarray) representing
            the reflection vector (h, k, l).
        sf (np.ndarray, optional): A 1D NumPy array of shape (4,) containing the
        structure factor components [abs(U), phase(U), abs(UA), phase(UA)] for the
        input reflection `gg`. Defaults to None.

    Returns:
        list[tuple] | tuple[list[tuple], np.ndarray]:
            - If `sf` is None, returns a list of tuples, where each tuple is a
              symmetry-equivalent reflection (h, k, l).
            - If `sf` is not None, returns a tuple `(ggs, sfs)`, where `ggs` is
              the list of equivalent reflections and `sfs` is a 2D NumPy array
              of shape `(n_equivalent, 4)` containing the transformed structure
              factors.

    Raises:
        ValueError: If the `spgr.gp` symmetry parameters are not initialized or
            if the input `gg` is not a 1D sequence.
    """
    #check gg type
    if not np.any(spgr.gp):
        raise ValueError('gp is not initialized, run simulation first to initialize values in common block')
    if not isinstance(gg, np.ndarray):
        gg = np.array(gg)
    if gg.ndim != 1:
        raise ValueError('gg must be 1D sequence')
    
    gp = spgr.gp
    ngp = spgr.ngp
    hp = np.zeros((3,ngp))    
    # TRP = np.sum(gg*gp[:ngp,9:12], axis=1)
    hp[0,:ngp] = np.sum(gp[:ngp,0:9:3]*gg, axis=1)
    hp[1,:ngp] = np.sum(gp[:ngp,1:9:3]*gg, axis=1)
    hp[2,:ngp] = np.sum(gp[:ngp,2:9:3]*gg, axis=1)
    unique_vals, unique_indices = np.unique(hp.astype(np.int16).T, axis=0, return_index=True)
    ggs = list(map(tuple, unique_vals))
    if sf is None: return ggs
    xf = sf[0]*np.exp(1j*np.deg2rad(sf[1]))
    af = sf[2]*np.exp(1j*np.deg2rad(sf[3]))
    sfs = np.zeros((len(ggs), 4), dtype=np.float32)
    for idx, (vals, ind) in enumerate(zip(unique_vals, unique_indices)):
        phase = np.exp(-2*np.pi*1j*np.dot(gp[ind,9:], gg))
        curr_xf = xf*phase
        curr_af = af*phase
        sfs[idx] = [np.abs(curr_xf), np.angle(curr_xf, deg=True), np.abs(curr_af), np.angle(curr_af, deg=True)]

    return ggs, sfs



def getSymmetry():
    """Retrieves the active symmetry operations.

    Returns:
        np.ndarray: A 2D NumPy array containing the symmetry operations
        (rotation matrices and translation vectors) from the `spgr` common block.
    """
    return spgr.gp[:spgr.ngp,:]