from matplotlib import pyplot as plt
import numpy as np

def intensity(field,cmap='viridis'):
    if field.ndim == 3:
        field = np.linalg.norm(field,axis=2)
    field = np.abs(field)**2

    plt.figure().frameon = False
    plt.axis('off')
    plt.imshow(field,cmap, vmin=0, vmax=max(1e-5,np.max(field)))


def phase(field):
    return np.angle(field)

def power(x,y,field):
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    if field.ndim == 3:
        field = np.linalg.norm(field,axis=2)

    power = np.sum(np.abs(field)**2) * dx * dy
    return power


def polarization(x,y,field, t=0, pace=20,scale=None):
    Ex = np.real(field[...,0] * np.exp(-1j*t))
    Ey = np.real(field[...,1] * np.exp(-1j*t))

    plt.figure()
    plt.quiver(x[::pace, ::pace], y[::pace, ::pace],
               Ex[::pace, ::pace],Ey[::pace, ::pace],
               scale=scale)
    
    plt.axis('equal')
    plt.axis('off')
    plt.show()
