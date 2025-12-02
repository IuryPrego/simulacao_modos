import numpy as np


class OpticalField:
    def __init__(self, x, y, z=0, wavelength=632.8e-9, E=None):
        self.x = x
        self.y = y
        self.z = z
        self.dx = float(x[0, 1] - x[0, 0])
        self.dy = float(y[1, 0] - y[0, 0])
        self._lambda_ = wavelength
        self.k = 2 * np.pi / wavelength
        if np.sum(E) != None:
            self.E = E
        else:
            self.E = np.zeros_like(x,dtype='complex128')



    def intensity(self):
        return np.abs(self.E)**2


    def phase(self):
        return np.angle(self.E)


    def normalize(self):
        dx, dy = self.dx,self.dy
        norm = np.sum(np.abs(self.E) ** 2) * dx * dy
        if norm != 0:
            self.E /= np.sqrt(norm)
        else:
            raise ValueError('Can`t normalize empty field')

