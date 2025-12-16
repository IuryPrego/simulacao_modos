from matplotlib import pyplot as plt
import numpy as np

def intensity(E,cmap='viridis'):
    if E.ndim == 3:
        E = np.linalg.norm(E,axis=2)
    E = np.abs(E)**2
    plt.imshow(E,cmap, vmin=0, vmax=max(1e-5,np.max(E)))


def phase(E):
    return np.angle(E)

def power(x,y,E):
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    power = np.sum(np.abs(E))** 2 * dx * dy
    return power
