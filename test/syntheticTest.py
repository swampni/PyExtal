

import numpy as np
import matplotlib.pyplot as plt


from dinfo import LARBEDDiffractionInfo
from roi import LARBEDROI
from callBloch import lookupSF, terminate
from optimize import FineOptimize
from gof import Chi2_poisson

import mpi4py.MPI as MPI

plt.rcParams['image.cmap'] = 'inferno'
dp = np.load("../box/syntheticData/Si110/noisy1e5_cbed.npy")
dp[dp < 1] = 1
gindex = np.load("../box/syntheticData/Si110/indices.npy")
dtpar = [0,0,0,0,0]
background = 0

dinfo = LARBEDDiffractionInfo(dp,  384,  0, 0, 127/3.4, '../examples/Si_undoped/si110.dat', dtpar, background,gindex)
rotation = 0
dpCenter = [127, 127]
dpSize = [255,255]
gInclude = [(0,0,0),(-1,1,-1), (1,-1,-1), (-1,1,1),(1,-1,1),(0,0,-4), (0,0,4), (-2,2,0),(2,-2,0), (-1,1,-3),(1,-1,-3),(-1,1,3),(1,-1,3),(-3,3,-1),(3,-3,-1),(-3,3,1),(3,-3,1), (-2,2,-2),(2,-2,-2),(-2,2,2),(2,-2,2)]

roi = LARBEDROI(dinfo=dinfo, rotation=rotation, gx=np.array([0,0,2]), gInclude=gInclude, dpCenter=dpCenter)

reflection_list = [(1,-1,1), (-2,2,0), (0,0,4), (-1,1,3), (2,-2,2), (3,-3,1)]
roi.selectROI(np.array([[[-60,-60], [-60, 60], [60,-60], [40,40]]]))

fine = FineOptimize(dinfo=dinfo, roi=roi, reflections=reflection_list, errorFunc=Chi2_poisson(), symUpdate=True)
temp = lookupSF(reflection_list)
temp[4] = np.array([0.0005,0,1e-36,180])
fine.getx0(temp[:,::2])
tempRange = np.array([temp*0.95, temp*1.05])
fine.getRange(tempRange[:,:,::2])
fine.display([10,20,30])
fine.optimize()
fine.display([10,20,30])
print(fine.history[0].final_simplex)
input()
terminate()
MPI.Finalize()
