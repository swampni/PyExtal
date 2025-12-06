import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import farid

from pyextal.dinfo import CBEDDiffractionInfo, LARBEDDiffractionInfo
from pyextal.roi import CBEDROI, LARBEDROI
from pyextal.optimize import CoarseOptimize, FineOptimize
from pyextal.gof import Chi2_LARBED
from pyextal.symmetry import appliedSymmetry
import pyextal.blochwave as bw

import mpi4py.MPI as MPI


plt.rcParams['image.cmap'] = 'inferno'
dp = np.load("Larbed/Si_undoped/111_110sys/region1/Store_deconv.npy")
dp = np.flip(dp,axis=2)
gindex = np.load("Larbed/Si_undoped/111_110sys/region1/g_vectorsSi.npy")
variance = np.load("Larbed/Si_undoped/111_110sys/region1/Store_variance.npy")
variance = np.flip(variance,axis=2)

dtpar = [0,0,0,0,0]
background = 0

dp[dp < 1] = 1

dinfo = LARBEDDiffractionInfo(dp,  900,  0, 0, 31.78, 'examples/Si_undoped/si110.dat', dtpar, background,gindex, varianceMaps=variance)

rotation = -112.07+35.1+180
nthgx = [0,0,0]
# dpCenter = [153, 160]
dpCenter = [127, 127]
# dpSize = [256,256]
dpSize = [255,255]
dp_index = 0
sim_index = 0
gInclude = [(0,0,0),(2,-2,0), (-2,2,0)]

roi = LARBEDROI(dinfo=dinfo, rotation=rotation, gx=np.array([1,-1,0]), gInclude=gInclude) 

# print(dinfo.getSF((2,2,0)))
# print(appliedSymmetry((1,1,1), dinfo.getSF((1,1,1))))
print(bw.constants.buildversion())
# for beam in dinfo.symmetry.beamGroup[0]:
    
#     print(beam, dinfo.getSF(beam))
# print(dinfo.getSF((1,-1,-1)))
# print(dinfo.getSF((-1,1,1)))


# roi.selectROI(np.array([[[50,80], [50, 230], [220,80], [150,170]]]))


# coarse = CoarseOptimize(datpath='../examples/Si_undoped/si111_110sys.dat', dinfo=dinfo, roi=roi)

MPI.Finalize()