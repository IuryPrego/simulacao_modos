import numpy as np

def linear_polarized(field,method='linear',theta=0):
    match method:
        case 'linear':
            phase_mat = np.array([[np.cos(theta),-np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
        case 'halfwave':
            phase_mat = np.array([[np.cos(2*theta),  np.sin(2*theta)],
                                  [np.sin(2*theta), -np.cos(2*theta)]])

    if field.ndim == 3:
        return phase_mat*field
    else:
        P = np.ones([2,*field.shape])
        P = phase_mat*P
        return field[..., None] * P