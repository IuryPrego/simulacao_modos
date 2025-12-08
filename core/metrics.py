import numpy as np

def Intensity(E):
    if E.ndim == 3:
        E = np.linalg.norm(E,axis=0)
        print(E.shape)
    return np.abs(E)**2

def Phase(self):
    return np.angle(self.E)

def Power(x,y,E):
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    power = np.sum(np.abs(E))** 2 * dx * dy
    return power
