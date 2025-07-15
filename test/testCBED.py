import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import farid
from pyextal.dinfo import CBEDDiffractionInfo
from pyextal.roi import CBEDROI
from pyextal.optimize import CoarseOptimize, FineOptimize
from pyextal.gof import Chi2, Chi2_const
from pyextal.callBloch import terminate

plt.rcParams['image.cmap'] = 'inferno'
data = np.fromfile("box/z200n.img", dtype=np.int16, offset=8).reshape(-1, 1024,1024)
dp = data[0]
mtf = np.load('box/YAG 120 CCD MTF.npy')



dtpar = [16.868, 4.1797e-5, 1.1108, 1.047, 0.07229]
background = 38.0681229

dinfo = CBEDDiffractionInfo(dp, 775.9,  0.3, 0, 158.708, 'examples/Cu2O/Cu2O.dat', dtpar, mtf, background,)


rotation = -19.44
dpCenter = [250, 950]
dpSize = [500,1200]

roi = CBEDROI(dinfo=dinfo, rotation=rotation, gx=np.array([1,0,0]), gInclude=[(0,0,0),(-2,0,0),(-4,0,0)], dpCenter=dpCenter, dpSize=dpSize) 


terminate()