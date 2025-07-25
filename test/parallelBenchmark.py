import time
from callBloch import simulate, calibrateLARBED, LARBED
import numpy as np
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI

# time the simulation with different number of processors


def sim(ncores):
    simulate('examples/Cu2O/Cu2O_LARBED.dat', 775, [500,  350, 157.8, 1200, 1200, 0], 0,0.3, ncores=ncores)

cores = [1, 2, 4, 8, 11,12]
iterations = 5

calTime = np.zeros((len(cores), iterations))
for idx, ncores in enumerate(cores):
    for i in range(iterations):
        t0 = time.time()
        sim(ncores)
        t1 = time.time()
        calTime[idx, i] = t1-t0
        print('Time to simulate LARBED pattern with {} cores: {} s'.format(ncores, t1-t0))

np.save('calTime.npy', calTime)
# plot the results
fig, ax = plt.subplots()
ax.plot(cores, calTime.mean(axis=1))
ax.errorbar(cores, calTime.mean(axis=1), yerr=calTime.std(axis=1), fmt='o')
ax.set_xlabel('Number of cores')
ax.set_ylabel('Time (s)')
ax.set_title('Time to simulate eigenvectors, eigenvalues and inverse')
plt.show()



