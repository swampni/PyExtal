"""This module provides functions for generating and visualizing 3D electron density maps.

It includes capabilities for Fourier synthesis from structure factors, plotting 3D
isosurfaces and 2D slices of the density map, and writing the volumetric data to
a Gaussian cube file for visualization in other software like VESTA.
"""
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

from pyextal.dinfo import BaseDiffractionInfo, LARBEDDiffractionInfo
from pyextal.roi import LARBEDROI
from pyextal.blochwave import defaultsf, constants, cryst, difpar, gram


def fourierSynthesis(getSF:callable, order:int, sampling:int) -> np.ndarray:
    """Performs Fourier synthesis to generate a 3D density map.

    This function constructs a 3D electron density map by summing crystallographic
    structure factors over a specified range of Miller indices.

    Args:
        getSF (callable): A function that takes a Miller index tuple (h, k, l)
            and returns its complex structure factor.
        order (int): The maximum Miller index (h, k, l) to include in the synthesis.
            The range will be from -order to +order for each index.
        sampling (int): The number of grid points along each dimension of the
            output density map.

    Returns:
        np.ndarray: A 3D real-valued density map, normalized by the unit cell volume.
    """
    
    densityMap = np.zeros((sampling,sampling,sampling), dtype=np.complex64)
    x, y, z = np.mgrid[:sampling,:sampling,:sampling] / sampling
    for h in range(-order,order+1,1):
        for k in range(-order,order+1,1):
            for l in range(-order,order+1,1):
                if h == 0 and k == 0 and l == 0:continue
                # if h**2 + k**2 + l**2 > 16: continue
                sf = getSF((h,k,l))
                
                densityMap += (sf[0] + 1j*sf[1]) *np.exp(-2*np.pi*1j*(h*x+k*y+l*z))   
    return densityMap.real / constants.vol

def plotDensityMapIsosurface(density_map: np.ndarray, threshold:float=10) -> None:
    """Plots a 3D isosurface of the density map.

    Uses the marching cubes algorithm to extract and display a surface at a given
    density threshold. Note: The orientation of X, Y, Z axes might need to be
    verified with software like VESTA.

    Args:
        density_map (np.ndarray): The 3D density map to be visualized.
        threshold (float, optional): The density value at which to draw the
            isosurface. Defaults to 10.
    """
    # Extract the surface using marching cubes
    verts, faces, _, _ = marching_cubes(density_map, level=threshold)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, facecolor='blue', shade=True)
    
    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')    
    plt.show()

def plotDensityMap_2D(density_map: np.ndarray, clim:tuple, cmap:str)-> None:
    """Plots 2D slices of the density map.

    Displays four 2D slices taken from the 3D density map along one axis.

    Args:
        density_map (np.ndarray): The 3D density map.
        clim (tuple): A tuple (min, max) to set the color limits.
        cmap (str): The colormap to use for the plot.
    """
    fig, axes = plt.subplots(nrows=1, ncols=4)
    for i in range(4):    
        axes[i].imshow(density_map[i*16,:,:], vmin=clim[0], vmax=clim[1], cmap=cmap)
        print(np.max(density_map[i*16,:,:]))        
    plt.show()

def write_cube(filename:str, data:np.ndarray) -> None:
    """Writes volumetric data to a Gaussian cube file.

    This function creates a .cube file, which is a standard format for volumetric
    data like electron density, readable by visualization software (e.g., VESTA).
    The atom and cell information is read from the global `cryst` object.

    Args:
        filename (str): The path to the output .cube file.
        data (np.ndarray): A 3D NumPy array containing the charge density values.
    """
    nx, ny, nz = data.shape
    
    with open(filename, "w") as f:
        f.write("Generated by pyextal\n")
        f.write("Charge density data\n")
        
        # Number of atoms and grid origin
        f.write(f"{cryst.natoms:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        
        # Grid dimensions and voxel spacing TODO: non-cubic system
        f.write(f"{nx:5d} {cryst.cell[0]/nx:12.6f} 0.000000 0.000000\n")
        f.write(f"{ny:5d} 0.000000 {cryst.cell[1]/ny:12.6f} 0.000000\n")
        f.write(f"{nz:5d} 0.000000 0.000000 {cryst.cell[2]/nz:12.6f}\n")
        # Atomic positions
        for idx, (x, y, z) in enumerate(cryst.atpar.T[:cryst.natoms,:3]):
            f.write(f"{int(cryst.zt[int(cryst.itype[idx])-1]+0.5):5d} 0.000000 {x:12.6f} {y:12.6f} {z:12.6f}\n")
        
        # Charge density values
        data_flat = data.ravel()
        for i, value in enumerate(data_flat):
            f.write(f"{value:12.5E} ")
            if (i + 1) % 6 == 0:  # 6 values per line
                f.write("\n")





