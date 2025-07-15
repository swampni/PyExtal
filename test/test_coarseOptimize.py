import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
import matplotlib.pyplot as plt



from pyextal.dinfo import LARBEDDiffractionInfo
from pyextal.roi import LARBEDROI
from pyextal.optimize import CoarseOptimize


def fake_LARBED(param, thickness, **kwargs):
    if thickness > 499.999 and thickness < 500.001:
        fake_sim = np.ones((1, 100, 100))
        fake_sim[0,10:,:] = -1
        fake_sim[0, :, 10:] = -1
        
        marker = np.zeros((10, 10))
        marker[4:6,:] = 1
        marker[:, 4:6] = 1
        marker[0, 2:8] = 1
        marker[2:8, 0] = 1
        fake_sim[0, :10, :10] = marker

        return np.roll(fake_sim, shift=(10,20), axis=(1,2))  # Shift the array to simulate a diffraction pattern
    else:
        return np.ones((1, 100, 100))

def fake_calibrateLARBED(param, gl):
    return 100, 1



def test_coarse_optimize(monkeypatch):
    import pyextal.optimize
    monkeypatch.setattr(pyextal.optimize, 'LARBED', fake_LARBED)
    monkeypatch.setattr(pyextal.optimize, 'calibrateLARBED', fake_calibrateLARBED)

    fake_dp = np.arange(10000).reshape(100, 100)
    marker = np.zeros((10, 10))
    marker[4:6,:] = 20000
    marker[:, 4:6] = 20000
    marker[0, 2:8] = 20000
    marker[2:8, 0] = 20000
            
    fake_dp[:10, :10] = marker    
    fake_dp = fake_dp[np.newaxis, :, :]   # Add a new axis to simulate a 3D array
    fake_variance = np.arange(10000).reshape(100, 100).T
    fake_variance = fake_variance[np.newaxis, :, :]   # Add a new axis to simulate a 3D array
    gindex = np.array([[0, 0, 0]])

    dinfo = LARBEDDiffractionInfo(fake_dp, 505, 0, 0, 50, 'test/test_data/parseTest.dat', gindex, fake_variance)

    # test with no rotation
    roi = LARBEDROI(dinfo=dinfo, rotation=0, gx=np.array([0, 0, 4]), gInclude=[(0,0,0)])
    roi.selectROI(np.array([[[0, 0], [0, 9], [9, 0], [10, 10]]]))

    
    coarse = CoarseOptimize(dinfo=dinfo, roi=roi)
    coarse.optimizeOrientationThickness()


    
    pyextal.callBloch.terminate()

    assert coarse.thickness > 499.999 and coarse.thickness < 500.001
    assert coarse.loc == (10, 20)

    assert dinfo.tiltX == -20
    assert dinfo.tiltY == 10

