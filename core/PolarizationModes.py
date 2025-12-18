import numpy as np


def scalar_to_vector(field, pol=(1,0)):
    pol = np.asarray(pol, dtype=complex)
    pol = pol / np.linalg.norm(pol)
    return field[...,None] * pol

def optical_element(field,theta=0,method='rotation'):
    match method:
        case 'rotation':
            cos, sen = np.cos(theta), np.sin(theta)
            phase_mat = np.array([[ cos,-sen],
                                  [ sen, cos]])
        case 'half':
            cos, sen = np.cos(2*theta), np.sin(2*theta)
            phase_mat = np.array([[cos, sen],
                                  [sen,-cos]])
        case 'quarter':
            cos, sen = np.cos(theta), np.sin(theta)
            phase_mat = np.exp(-1j*np.pi/4) * np.array([[cos**2 + 1j*sen**2, (1-1j)*cos*sen],
                                                        [(1-1j)*cos*sen, sen**2 + 1j*cos**2]])
        case _:
            raise TypeError("method must be 'half', 'quarter'")
        
    if field.ndim == 3:
        return field@phase_mat.T
    else:
        raise ValueError('optical_element expects a np.array with a shape: (..., 2).')
    
def filter_polarizer(field,theta=0,method='linear'):
    match method:
        case 'linear':
            cos, sen = np.cos(theta), np.sin(theta)
            phase_mat = np.array([[cos**2,cos*sen],
                                  [cos*sen, sen**2]])
        case 'Rcirc':
            phase_mat = 1/2*np.array([[1,  -1j],
                                      [1j, 1]])
        case 'Lcirc':
            phase_mat = 1/2*np.array([[1,  1j],
                                      [-1j, 1]])
        case _:  
            raise TypeError("method must be 'linear', 'Rcirc' or 'Lcirc'")

    if field.ndim == 3:
        return field@phase_mat.T
    else:
        raise ValueError('filter_polarizer expects a np.array with a shape: (..., 2).')
    