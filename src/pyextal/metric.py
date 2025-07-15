import numpy as np

def sumx(gram, v1, v2):
    '''
    calculate the inner product with metrical matrix
    '''
    return np.dot(np.dot(v1, gram), v2.T)

def volume(gram):
    '''
    calculate the volume of the unit cell
    '''
    return np.sqrt(np.linalg.det(gram))


def angle(gram, v1, v2):
    '''
    calculate the angle between two vectors
    '''
    return np.arccos(sumx(gram, v1, v2)/np.sqrt(sumx(gram, v1, v1)*sumx(gram, v2, v2)))


def scale(gram, v):
    return sumx(gram,v,v)**0.5
