import numpy as np
from scipy.special import genlaguerre, hermite
from Core.Field import OpticalField
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
    def __init__(self,p, l, x, y, z=0, w0=1e-3, wavelength=632.8e-9,thetax=0,thetay=0):
        super().__init__(x, y, z, w0, wavelength)
        self.p = p
        self.l = l
        self.thetax = thetax
        self.thetay = thetay
        self.generate()
        self.normalize()


    def w(self):
        return self.w0 * np.sqrt(1 + (self.z / self.zR)**2)


    def generate(self):
        r = np.sqrt(self.x**2 + self.y**2)
        phi = np.arctan2(self.y, self.x)
        z = self.z

        k = self.k
        l,p = self.l, self.p
        w = self.w()
        w0 = self.w0
        L = genlaguerre(p, abs(l))(2*r**2/w**2)

        amplitude = (w0 / w) * ((np.sqrt(2) * r / w)**abs(l)) * L * np.exp(-r**2 / w**2)
        gouy_phase = self.__gouy__()
        
        if z != 0:
            phase = k * z + k * r**2 / (2 * self.__R__()) - gouy_phase + l * phi
        else:
            phase = - gouy_phase + l * phi

        self.E = amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(self.thetax)*self.x) * np.exp(1j*k*np.sin(self.thetay)*self.y)
        
    def intensity(self):
        return np.abs(self.E)**2


    def normalize(self):
        norm = np.sqrt(np.sum(self.intensity()))
        self.E = self.E / norm

    def __R__(self):
        return self.z * (1 + (self.zR / self.z)**2)


    def __gouy__(self):
        return (2*self.p + abs(self.l) + 1) * np.arctan(self.z / self.zR)


class HermiteGaussMode(SpatialMode):
    def __init__(self, m, n, x, y, z=0, w0=1e-3, wavelength=632.8e-9,thetax=0,thetay=0):
        super().__init__(x, y, z, w0, wavelength)
        self.m = m
        self.n = n
        self.thetax = thetax
        self.thetay = thetay
        self.generate()
        self.normalize()


    def w(self):
        return self.w0 * np.sqrt(1 + (self.z / self.zR)**2)


    def generate(self):
        x,y,z = self.x,self.y,self.z
        w = self.w()
        m,n = self.m,self.n
        k = self.k

        H_n = hermite(n)(np.sqrt(2)*x/w)
        H_m = hermite(m)(np.sqrt(2)*y/w)

        amplitude = H_n*np.exp(-x**2/w**2)*H_m*np.exp(-y**2/w**2)
        gouy_phase = self.__gouy__()
        
        if z != 0:
            phase = k * z + k * (x**2+y**2) / (2 * self.__R__()) - gouy_phase
        else:
            phase = self.k * self.z - gouy_phase

        self.E = amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(self.thetax)*self.x) * np.exp(1j*k*np.sin(self.thetay)*self.y)
        
    def intensity(self):
        return np.abs(self.E)**2


    def normalize(self):
        norm = np.sqrt(np.sum(self.intensity()))
        self.E = self.E / norm

    def __R__(self):
        return self.z * (1 + (self.zR / self.z)**2)


    def __gouy__(self):
        return (self.m + self.n + 1) * np.arctan(self.z / self.zR)
