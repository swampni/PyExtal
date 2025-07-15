import numpy as np
import matplotlib.pyplot as plt

from pyextal.callBloch import bloch_parse, terminate, bloch, LARBED, simulate


param, hkl = bloch("examples/YIG/YIG.dat", ncores=8, subaper=0, HKL=True)
# print(param)

for beam in hkl:
    print(beam)
thickness = 600
dp = LARBED(param, thickness)



fig, axes = plt.subplots(1, 2)
axes[0].imshow(dp[0], cmap='inferno')
axes[1].imshow(dp[6], cmap='inferno')
plt.show()
terminate()
