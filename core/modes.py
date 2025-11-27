import numpy as np
from scipy.special import genlaguerre, hermite
from core.field import OpticalField
from abc import ABC, abstractmethod

class SpatialMode(OpticalField, ABC):

    def __init__(self, x, y, z, w0, wavelength):
        super().__init__(x, y, z, wavelength)
        self.w0 = w0
        self.zR = np.pi * w0**2 / wavelength

    @abstractmethod
    def generate(self):
        pass


class LaguerreGaussMode(SpatialMode):
    def __init__(self,p, l, x, y, z=0, w0=1e-3, wavelength=632.8e-9):
        super().__init__(x, y, z, w0, wavelength)
        self.p = p
        self.l = l
        self.generate()


    def w(self):
        return self.w0 * np.sqrt(1 + (self.z / self.zR)**2)


    def R(self):
        return self.z * (1 + (self.zR / self.z)**2)


    def gouy(self):
        return -(2*self.p + abs(self.l) + 1) * np.arctan(self.z / self.zR)


    def generate(self):
        r = np.sqrt(self.x**2 + self.y**2)
        phi = np.arctan2(self.y, self.x)

        w = self.w()
        
        L = genlaguerre(self.p, abs(self.l))(2*r**2/w**2)
        amplitude = (self.w0 / w) * ((np.sqrt(2) * r / w)**abs(self.l)) * L * np.exp(-r**2 / w**2)
        gouy_phase = self.gouy()
        
        if self.z != 0:
            R = self.R()
            phase = self.k * self.z + self.k * r**2 / (2 * R) + gouy_phase + self.l * phi
        else:
            phase = self.k * self.z + gouy_phase + self.l * phi

        self.E = amplitude * np.exp(1j * phase)
        
    def intensity(self):
        return np.abs(self.E)**2


    def normalize(self):
        norm = np.sqrt(np.sum(self.intensity()))
        self.E = self.E / norm
