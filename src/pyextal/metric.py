"""Metric Tensor Calculations.

This module provides functions for performing crystallographic calculations using a
Gram matrix (metric tensor). These functions are essential for working with
crystal lattices where the basis vectors are not necessarily orthogonal.
"""
import numpy as np

def sumx(gram: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the inner product of two vectors using a Gram matrix.

    This function computes v2^T * G * v1, where G is the Gram matrix.

    Args:
        gram (np.ndarray): The Gram matrix (metric tensor).
        v1 (np.ndarray): The first vector.
        v2 (np.ndarray): The second vector.

    Returns:
        float: The inner product of the two vectors.
    """
    return np.dot(np.dot(v1, gram), v2.T)

def volume(gram: np.ndarray) -> float:
    """Calculates the volume of the unit cell from its Gram matrix.

    The volume is the square root of the determinant of the Gram matrix.

    Args:
        gram (np.ndarray): The Gram matrix of the unit cell.

    Returns:
        float: The volume of the unit cell.
    """
    return np.sqrt(np.linalg.det(gram))


def angle(gram: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the angle between two vectors in a space defined by a Gram matrix.

    Args:
        gram (np.ndarray): The Gram matrix.
        v1 (np.ndarray): The first vector.
        v2 (np.ndarray): The second vector.

    Returns:
        float: The angle between the vectors in radians.
    """
    return np.arccos(sumx(gram, v1, v2)/np.sqrt(sumx(gram, v1, v1)*sumx(gram, v2, v2)))


def scale(gram: np.ndarray, v: np.ndarray) -> float:
    """Calculates the magnitude (norm) of a vector using a Gram matrix.

    Args:
        gram (np.ndarray): The Gram matrix.
        v (np.ndarray): The vector whose magnitude is to be calculated.

    Returns:
        float: The magnitude of the vector.
    """
    return sumx(gram,v,v)**0.5
