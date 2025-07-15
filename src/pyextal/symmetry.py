from collections import OrderedDict, defaultdict

import numpy as np 

from pyextal.blochwave import spgr


class GroupSymmetry:
    '''
    class for grouping symmetry equivalent beams

    args:
            beams: list of (h,k,l) included
        
    '''
    def __init__(self, beams):
        '''
            tabulate the symmetry equivalent beams
            args:
                beams: list of (h,k,l) included
        '''
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
                invBeam = tuple(-Ugh_table[i,j])
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
                if invBeam not in record:                    
                    # not included in any group, add a new group
                    allSym = appliedSymmetry(invBeam)
                    self.beamGroup.append(allSym)
                    igroup = len(self.beamGroup) - 1
                    # assign all symmetry equivalent beams to the same group
                    for symBeam in allSym:
                        self.beam2Group[symBeam] = igroup
                        record.add(symBeam)

                currentGroup = self.getGroup(beam)
                index = currentGroup.index(beam)

                # calculate the phase factor for symmetry related beams, 
                # see Waser et al. (1955), Acta Cryst. DOI:10.1107/S0365110X55001862
                phase = np.exp(-2*np.pi*1j*np.dot(self.gp[index, 9:], currentGroup[0]))
                self.phaseDict[beam] = phase
                self.Ugh_dict[self.beam2Group[beam]].append((i,j,phase))
                
                # do the same for the inverse beam
                currentGroup = self.getGroup(invBeam)
                index = currentGroup.index(invBeam)
                phase = np.exp(-2*np.pi*1j*np.dot(self.gp[index, 9:], currentGroup[0]))
                self.phaseDict[invBeam] = phase
                
                self.Ugh_dict[self.beam2Group[invBeam]].append((i,j,phase))
        print('group symmetry initialized', flush=True)
    
    def getGroup(self, beam):
        '''return the group of the beam'''
        return self.beamGroup[self.beam2Group[beam]]
    
    def getPos(self, beam):
        '''return the equivalent beams positions in Ugh matrix'''        
        beam = tuple(beam)
        currentPhase = self.phaseDict[beam]
        for i,j,phase in self.Ugh_dict[self.beam2Group[beam]]:
            yield i,j,phase/currentPhase


def appliedSymmetry(gg, sf=None):
    """
    Calculate the symmetry-equivalent reflections for a given reflection vector.
    This function finds and returns all equivalent reflections based on symmetry operations.
    It first checks if the symmetry parameters (spgr.gp and spgr.ngp) are initialized by
    ensuring that spgr.gp contains non-zero data. Then, it verifies that the provided reflection
    vector gg is a one-dimensional array; if not, it attempts to convert it into one. The function
    calculates a transformed reflection vector using the symmetry operations stored in the
    spgr.gp matrix and returns an integer matrix with each row corresponding to an equivalent reflection.
    Parameters:
        gg (array_like): A 1D sequence representing the reflection vector. It can be a list, tuple, or np.ndarray.
                         If not an np.ndarray, it will be converted to one.
        sf (np.ndarray): A 2D numpy array of shape (4) containing the structure factors for the current reflection
    Returns:
        ggs: (np.ndarray): A 2D numpy array of shape (ngp, 3) containing the integer reflection vectors
                    corresponding to each symmetry operation.
        sfs: (np.ndarray): A 2D numpy array of shape (ngp, 4) containing the structure factors for the symmetry equivalent reflections
        
    Raises:
        ValueError: If the symmetry parameters (spgr.gp) are not initialized.
        ValueError: If the input gg is not a 1-dimensional sequence.
    """
    '''
    find all equivalent reflections for given symmetry, need to be initialized (run simulation first to initialize values in common block)

    '''
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
    return spgr.gp[:spgr.ngp,:]