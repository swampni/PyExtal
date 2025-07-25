from callBloch import bloch_parse, bloch_run, updateSymUgh, LARBED,terminate
from box.structureFactorCalculator import SFcalculator
import numpy as np
import matplotlib.pyplot as plt

import mpi4py.MPI as MPI
import matplotlib
param = bloch_parse(fname='examples/Si_undoped/si111sys.dat')

reflections = [(1,-1,1), (-2,2,0), (0,0,4), (-1,1,3), (2,-2,2), (3,-3,1)]
value = np.array([
[0.0542785,5.6254957e-04],
[0.0460924, 7.3504535e-04],
[0.0273639, 6.6099927e-04],
[0.0262070, 4.9703574e-04],
[0.0005406, 0],
[0.0165972, 4.5006312e-04]])

updateSymUgh(reflections=reflections, values=value)
param, hkl = bloch_run(param, ncores=8, HKL=True)
cbed = LARBED(param, 600, )


vec1 = np.array([0,0,1])
vec2 = np.array([1,-1,0])/np.linalg.norm([1,-1,0])
dir1 = np.array([70,0])
dir2 = np.array([0,70])

# canvas = np.zeros((1000,1000))
# for hkl, dp in zip(param.hklout.T, cbed):
#     coff1 = hkl.dot(vec1)
#     coff2 = hkl.dot(vec2)
#     pos = coff1*dir1 + coff2*dir2
#     pos = pos.astype(np.int32)
#     canvas[500+pos[1]-40:500+pos[1]+41, 500+pos[0]-40:500+pos[0]+41] = np.fliplr(dp)
# plt.imshow(np.rot90(canvas, k=3), cmap='gray', vmin=0, vmax=0.5)
# plt.show()

# Add Poisson noise to CBED patterns
noisy_cbed = np.random.poisson(cbed * 1e5)


np.save('box/syntheticData/Si110_111sys/cbed.npy', cbed)
np.save('box/syntheticData/Si110_111sys/noisy1e5_cbed.npy', noisy_cbed)
np.save('box/syntheticData/Si110_111sys/indices.npy', param.hklout.T)
np.save('box/syntheticData/Si110_111sys/hklInclude.npy', hkl)



terminate()
MPI.Finalize()
