import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
import matplotlib.pyplot as plt
from skimage.transform import rotate


from pyextal.dinfo import LARBEDDiffractionInfo
from pyextal.roi import LARBEDROI

def test_sampling():
    fake_dp = np.arange(10000).reshape(100, 100)
    fake_dp = fake_dp[np.newaxis, :, :]   # Add a new axis to simulate a 3D array
    fake_variance = np.arange(10000).reshape(100, 100).T
    fake_variance = fake_variance[np.newaxis, :, :]   # Add a new axis to simulate a 3D array
    gindex = np.array([[0, 0, 0]])

    dinfo = LARBEDDiffractionInfo(fake_dp, 100, 0, 0, 1, 'test/test_data/parseTest.dat', gindex, fake_variance)

    # test with no rotation
    roi = LARBEDROI(dinfo=dinfo, rotation=0, gx=np.array([0, 0, 4]), gInclude=[(0,0,0)])
    roi.selectROI(np.array([[[0, 0], [0, 9], [9, 0], [10, 10]]]))
    assert_allclose(roi.templates[0], fake_dp[0, :10, :10].flatten(), rtol=5e-3)
    assert_allclose(roi.variance[0], fake_variance[0, :10, :10].flatten(), rtol=5e-3)

    # test with 90 degree rotation
    roi = LARBEDROI(dinfo=dinfo, rotation=90, gx=np.array([0, 0, 4]), gInclude=[(0,0,0)])
    roi.selectROI(np.array([[[1, 1], [1, 10], [10, 1], [10, 10]]]))
    assert_allclose(roi.templates[0], np.rot90(fake_dp[0], k=1)[1:11, 1:11].flatten(), rtol=5e-3)
    assert_allclose(roi.variance[0], np.rot90(fake_variance[0], k=1)[1:11, 1:11].flatten(), rtol=5e-3)

    # test with 180 degree rotation
    roi = LARBEDROI(dinfo=dinfo, rotation=180, gx=np.array([0, 0, 4]), gInclude=[(0,0,0)])
    roi.selectROI(np.array([[[1, 1], [1, 10], [10, 1], [10, 10]]]))
    assert_allclose(roi.templates[0], np.rot90(fake_dp[0], k=2)[1:11, 1:11].flatten(), rtol=5e-3)
    assert_allclose(roi.variance[0], np.rot90(fake_variance[0], k=2)[1:11, 1:11].flatten(), rtol=5e-3)

    # test with 270 degree rotation
    roi = LARBEDROI(dinfo=dinfo, rotation=270, gx=np.array([0, 0, 4]), gInclude=[(0,0,0)])
    roi.selectROI(np.array([[[1, 1], [1, 10], [10, 1], [10, 10]]]))
    assert_allclose(roi.templates[0], np.rot90(fake_dp[0], k=3)[1:11, 1:11].flatten(), rtol=5e-3)
    assert_allclose(roi.variance[0], np.rot90(fake_variance[0], k=3)[1:11, 1:11].flatten(), rtol=5e-3)

    # test with 45 degree rotation
    roi = LARBEDROI(dinfo=dinfo, rotation=45, gx=np.array([0, 0, 4]), gInclude=[(0,0,0)])
    roi.selectROI(np.array([[[65, 65], [65, 74], [74, 65], [10, 10]]]))
    rotated_dp = rotate(fake_dp[0], angle=45, resize=True, preserve_range=True)[65:75, 65:75]
    rotated_variance = rotate(fake_variance[0], angle=45, resize=True, preserve_range=True)[65:75, 65:75]    
    assert_allclose(roi.templates[0], rotated_dp.flatten(), rtol=5e-3)
    assert_allclose(roi.variance[0], rotated_variance.flatten(), rtol=5e-3)

    # test with 135 degree rotation
    roi = LARBEDROI(dinfo=dinfo, rotation=135, gx=np.array([0, 0, 4]), gInclude=[(0,0,0)])
    roi.selectROI(np.array([[[65, 65], [65, 74], [74, 65], [10, 10]]]))
    rotated_dp = rotate(fake_dp[0], angle=135, resize=True, preserve_range=True)[65:75, 65:75]
    rotated_variance = rotate(fake_variance[0], angle=135, resize=True, preserve_range=True)[65:75, 65:75]    
    assert_allclose(roi.templates[0], rotated_dp.flatten(), rtol=5e-3)
    assert_allclose(roi.variance[0], rotated_variance.flatten(), rtol=5e-3)

    
    

