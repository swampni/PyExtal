import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


import pyextal.callBloch as callBloch
import pyextal.blochwave as bw

def test_bloch_parse():
    param = callBloch.bloch_parse(r'test/test_data/parseTest.dat')
    assert isinstance(param, callBloch.SimParams)

    assert bw.cryst.cell[0] == pytest.approx(5.4307)
    assert bw.cryst.cell[1] == pytest.approx(5.4307)
    assert bw.cryst.cell[2] == pytest.approx(5.4307)
    assert bw.cryst.cell[3] == pytest.approx(90.0)
    assert bw.cryst.cell[4] == pytest.approx(90.0)
    assert bw.cryst.cell[5] == pytest.approx(90.0)

    assert bw.gram.gmx[0,0] == pytest.approx(5.4307**2)
    assert bw.gram.gmx[1,1] == pytest.approx(5.4307**2)
    assert bw.gram.gmx[2,2] == pytest.approx(5.4307**2)
    assert bw.gram.gmx[0,1] == pytest.approx(5.4307**2* np.cos(np.pi/2))
    assert bw.gram.gmx[0,2] == pytest.approx(5.4307**2* np.cos(np.pi/2))
    assert bw.gram.gmx[1,2] == pytest.approx(5.4307**2* np.cos(np.pi/2))

    assert bw.xtal.nsite == 1

    assert bw.cryst.v0 == 0
    assert bw.difpar.kv == 300

    assert bw.difpar.nbeams == 19
    assert bw.difpar.sgmax == pytest.approx(3.5)
    assert bw.difpar.sgmin == pytest.approx(0.15)
    assert bw.difpar.omgmx == pytest.approx(10.0)
    assert bw.difpar.abm == pytest.approx(0.0)

    beams = np.array([
        [ 0,  0,  0],
        [ 1, -1, -1],
        [-1,  1, -1],
        [ 1, -1,  1],
        [-1,  1,  1],
        [ 0,  0, -2],
        [ 0,  0,  2],
        [ 2, -2,  0],
        [-2,  2,  0],
        [ 1, -1, -3],
        [-1,  1, -3],
        [ 1, -1,  3],
        [-1,  1,  3],
        [ 2, -2, -2],
        [-2,  2, -2],
        [ 2, -2,  2],
        [-2,  2,  2],
        [ 0,  0, -4],
        [ 0,  0,  4]
    ]).T
    assert_array_equal(bw.difpar.hkl, beams)
    assert param.nx == 40

    assert_array_almost_equal(param.snorm / np.linalg.norm(param.snorm), np.array([1,1,0] / np.sqrt(2)))



def test_updateUgh():
    param = callBloch.bloch_parse(r'test/test_data/parseTest.dat')
    callBloch.updateUgh(reflections=[(1,-1,-1)], values=np.array([[10,0,10,0]]))
    
    assert bw.bloch_module.ughr[0,1] == 10
    assert bw.bloch_module.ughr[1,0] == 10
    assert bw.bloch_module.ughr[0,4] == 10
    assert bw.bloch_module.ughr[4,0] == 10
    assert bw.bloch_module.ughr[13,1] == 10
    assert bw.bloch_module.ughr[1,13] == 10
    assert bw.bloch_module.ughr[5,2] == 10
    assert bw.bloch_module.ughr[2,5] == 10
    assert bw.bloch_module.ughr[8,2] == 10
    assert bw.bloch_module.ughr[2,8] == 10
    assert bw.bloch_module.ughr[7,3] == 10
    assert bw.bloch_module.ughr[3,7] == 10
    assert bw.bloch_module.ughr[4,16] == 10
    assert bw.bloch_module.ughr[16,4] == 10
    assert bw.bloch_module.ughr[6,3] == 10
    assert bw.bloch_module.ughr[3,6] == 10
    assert bw.bloch_module.ughr[9,5] == 10
    assert bw.bloch_module.ughr[5,9] == 10
    assert bw.bloch_module.ughr[12,6] == 10
    assert bw.bloch_module.ughr[6,12] == 10
    assert bw.bloch_module.ughr[14,10] == 10
    assert bw.bloch_module.ughr[10,14] == 10
    assert bw.bloch_module.ughr[17,10] == 10
    assert bw.bloch_module.ughr[10,17] == 10
    assert bw.bloch_module.ughr[15,11] == 10
    assert bw.bloch_module.ughr[11,15] == 10
    assert bw.bloch_module.ughr[18,11] == 10
    assert bw.bloch_module.ughr[11,18] == 10
    
    # Check all other elements are not 10
    
    for i in range(bw.bloch_module.ughr.shape[0]): 
        for j in range(bw.bloch_module.ughr.shape[1]):
              if (i,j) not in [(13,1), (1,13), (5,2), (2,5), (8,2), (2,8), 
                              (7,3), (3,7), (9,5), (5,9), (12,6), (6,12),
                              (14,10), (10,14), (17,10), (10,17), (15,11),
                              (11,15), (17,12), (12,17), (0,1),(1,0), (0,4), 
                              (4,0), (6,3), (3,6), (4,16), (16,4), (18,11), (11,18)]:
                    print(i,j)
                    assert bw.bloch_module.ughr[i,j] != 10

    
    







    
