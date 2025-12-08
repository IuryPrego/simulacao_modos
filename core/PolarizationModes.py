import numpy as np


def LinearPolarized(field,method='linear',theta=0):
    P = np.ones([2,*field.shape])
    match method:
        case 'linear':
            phase_mat = np.array([[np.cos(theta),-np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
        case 'halfwave':
            phase_mat = np.array([[np.cos(2*theta),  np.sin(2*theta)],
                                  [np.sin(2*theta), -np.cos(2*theta)]])
    P = phase_mat*P
    E = np.einsum('xy,xyi->xyi',field,P)
    

