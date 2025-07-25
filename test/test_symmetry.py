import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


from pyextal.symmetry import GroupSymmetry, appliedSymmetry
import pyextal.callBloch as callBloch
import pyextal.blochwave as bw


def test_appliedSymmetry():
    param = callBloch.bloch_parse(r'test/test_data/parseTest.dat')  

    reflections= appliedSymmetry(np.array([1,1,1]))

    equiv = np.array([
        [ 1, -1, -1],
        [-1,  1, -1],
        [ 1, -1,  1],
        [-1,  1,  1],
        [ 1, 1, 1],
        [-1, -1, -1],
        [ 1, 1, -1],
        [-1,  -1, 1],])
    
    # Sort arrays and test if they contain exactly the same elements
    assert_array_equal(np.sort(reflections, axis=0), np.sort(equiv, axis=0))


def test_appliedSymmetry_noncentro():
    param = callBloch.bloch_parse(r'test/test_data/parseTest_nocentro.dat')
    reflections= appliedSymmetry(np.array([1,1,1]))

    equiv = np.array([
        [ 1, 1, 1],
        [1, -1, -1],
        [ -1, 1,  -1],
        [-1,  -1,  1],
        ])
    
    # Convert arrays to sets of tuples for comparison
    equiv_sorted = {tuple(row) for row in equiv}
    reflections_sorted = {tuple(row) for row in reflections}

    
    assert reflections_sorted == equiv_sorted



def test_group_symmetry():
    param = callBloch.bloch_parse(r'test/test_data/parseTest.dat')
    beams = np.array([
        [ 0,  0,  0],
        [ 1, -1, -1],
        [-1,  1, -1],
        [ 1, -1,  1],
        [-1,  1,  1],
        [ 0,  0, -2],
        [ 0,  0,  2],
        [ 1, -1, -3],
        [-1,  1, -3],
        [ 1, -1,  3],
        [-1,  1,  3],       
    ])

    group_symmetry = GroupSymmetry(beams)
    assert group_symmetry.centro is True
    assert len(group_symmetry.beamGroup) == 10


        
def test_group_symmetry_noncentro():
    param = callBloch.bloch_parse(r'test/test_data/parseTest_nocentro.dat')
    beams = np.array([
        [ 0,  0,  0],
        [ 1, -1, -1],
        [-1,  1, -1],
        [ 1, -1,  1],
        [-1,  1,  1],
        [ 0,  0, -2],
        [ 0,  0,  2],
        [ 1, -1, -3],
        [-1,  1, -3],
        [ 1, -1,  3],
        [-1,  1,  3],       
    ])

    group_symmetry = GroupSymmetry(beams)

    assert group_symmetry.centro is False
    assert len(group_symmetry.beamGroup) == 20

    plus = set()
    minus = set()
    diff_table = beams.reshape(-1,1,3) - beams.reshape(1,-1,3)
    for i in range(diff_table.shape[0]):
        for j in range(diff_table.shape[1]):
            if np.all(diff_table[i, j]==np.array([1,1,1])) or np.all(diff_table[i, j]==np.array([-1,-1,1])) or np.all(diff_table[i, j]==np.array([-1,1,-1])) or np.all(diff_table[i, j]==np.array([1,-1,-1])):
                plus.add((i, j))
                # print(diff_table[i, j], 'plus')
            elif np.all(diff_table[i, j]==np.array([-1,-1,-1])) or np.all(diff_table[i, j]==np.array([1,1,-1])) or np.all(diff_table[i, j]==np.array([1,-1,1])) or np.all(diff_table[i, j]==np.array([-1,1,1])):
                minus.add((i, j))
                # print(diff_table[i, j], 'minus')

    pos_set = {tuple(pos) for pos in group_symmetry.Ugh_dict[group_symmetry.beam2Group[(1,1,1)]]}
    assert pos_set == plus
    pos_set = {tuple(pos) for pos in group_symmetry.Ugh_dict[group_symmetry.beam2Group[(-1,-1,-1)]]}
    assert pos_set == minus