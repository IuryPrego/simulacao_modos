import numpy as np


class OpticalField:
    def __init__(self, x, y, z, wavelength, E=None):
        self.x = x
        self.y = y
        self.z = z
        self._lambda_ = wavelength
        self.k = 2 * np.pi / wavelength
        self.E = E


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
