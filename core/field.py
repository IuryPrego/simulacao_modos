import numpy as np


class OpticalField:
    def __init__(self, x, y, z=0, wavelength=632.8e-9, E=None):
        self.x = x
        self.y = y
        self.z = z
        self._lambda_ = wavelength
        self.k = 2 * np.pi / wavelength
        if np.sum(E) != None:
            self.E = E
        else:
            self.E = np.zeros_like(x)



    def intensity(self):
        return np.abs(self.E)**2


    def phase(self):
        return np.angle(self.E)


    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.E)**2))
        if norm != 0:
            self.E /= norm

    
'''
    def total_power(self):
        return np.sum(self.intensity()) * self.dx * self.dy


    def apply_loss(self, transmission):
        self.field *= np.sqrt(transmission)


    def set_power(self, power_watts):
        self.field *= np.sqrt(power_watts)
        self.power_watts = power_watts
'''
