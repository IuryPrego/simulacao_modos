from ctypes import FormatError  
import numpy as np


# get your scalar field and convert in a vector field with a give polarization
def scalar_to_vector(field, pol=(1,0)):
    pol = np.array(pol, dtype=complex)
    pol = pol / np.linalg.norm(pol)
    return np.copy(field)[...,None] * pol


# Get the field and simulate half wave plates and quartes wave plates
def polarizers(field,theta=0,method='half'):
    field = np.copy(field)

    if field.ndim<3:
        field = scalar_to_vector(field)

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
            raise TypeError("method must be 'half', 'quarter', 'rotation")
        
    return field@phase_mat.T
    


# simulate a polarizer
# it measures the componente of the choosen polarization base
def filter_polarizer(field,theta=0,method='linear'):
    field = np.copy(field)

    if field.ndim<3:
        raise TypeError("the field should be 3-dimensional, polarize it in first place with: 'optical_element(x,y,field)'")

    match method:
        case 'linear':
            cos, sen = np.cos(theta), np.sin(theta)
            phase_mat = np.array([[cos**2,cos*sen],
                                  [cos*sen, sen**2]])
        case 'Lcirc':
            phase_mat = 1/2*np.array([[1,  -1j],
                                      [1j, 1]])
        case 'Rcirc':
            phase_mat = 1/2*np.array([[1,  1j],
                                      [-1j, 1]])
        case _:  
            raise TypeError("method must be 'linear', 'Rcirc' or 'Lcirc'")

    return field@phase_mat.T
    

# Simulates a q-plate
# An optical element that does crazy things, it couples polarization to the transverse intensity profile
def q_plate(x,y,field,theta=0,delta=np.pi,q=1/2):
    field = np.copy(field)
    
    if field.ndim<3:
        field = scalar_to_vector(field)

    phase_mat = np.array([[np.exp(1j*delta/2),0],
                          [0,np.exp(-1j*delta/2)]])

    alpha = q*np.arctan2(y, x) - theta
    cos, sen = np.cos(-alpha), np.sin(-alpha)
    rot = np.array([[cos,-sen],
                    [sen, cos]])
    rot_minus = rot.transpose(2,3,0,1)[...]

    cos, sen = np.cos(alpha), np.sin(alpha)
    rot = np.array([[cos,-sen],
                    [sen, cos]])
    rot_plus = rot.transpose(2,3,0,1)[...]
    
    
    phase_mat = rot_minus@phase_mat@rot_plus
    field = phase_mat @ field[..., None]

    return field[...,0]
