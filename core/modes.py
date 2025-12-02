from tkinter import W
import numpy as np
from scipy.special import genlaguerre, hermite
from Core.Field import OpticalField
from abc import ABC, abstractmethod

class SpatialMode(OpticalField, ABC):

    def __init__(self, x, y, z, wavelength):
        super().__init__(x, y, z, wavelength)

    @abstractmethod
    def generate(self):
        pass


class LaguerreGaussMode(SpatialMode):
    def __init__(self,p, l, x, y, z=0, w0=1e-3, wavelength=632.8e-9,thetax=0,thetay=0):
        super().__init__(x, y, z, wavelength)
        self.w0 = w0
        self.p = p
        self.l = l
        self.thetax = thetax
        self.thetay = thetay
        self.zR = np.pi * w0**2 / wavelength
        self.generate()
        self.normalize()


    def generate(self):
        r = np.sqrt(self.x**2 + self.y**2)
        phi = np.arctan2(self.y, self.x)
        z = self.z

        k = self.k
        l,p = self.l, self.p
        w0 = self.w0
        w = OpticalFunctions.w(w0,z,self.zR)
        L = genlaguerre(p, abs(l))(2*r**2/w**2)

        amplitude = (w0 / w) * ((np.sqrt(2) * r / w)**abs(l)) * L * np.exp(-r**2 / w**2)
        gouy_phase = self.__gouy__()
        
        if z != 0:
            phase = k * z + k * r**2 / (2 * OpticalFunctions.R(z,self.zR)) - gouy_phase + l * phi
        else:
            phase = - gouy_phase + l * phi

        self.E = amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(self.thetax)*self.x) * np.exp(1j*k*np.sin(self.thetay)*self.y)
        
    def intensity(self):
        return np.abs(self.E)**2


    def normalize(self):
        norm = np.sqrt(np.sum(self.intensity()))
        self.E = self.E / norm


    def __gouy__(self):
        return (2*self.p + abs(self.l) + 1) * np.arctan(self.z / self.zR)


class HermiteGaussMode(SpatialMode):
    def __init__(self, m, n, x, y, z=0, w0=1e-3, wavelength=632.8e-9,thetax=0,thetay=0):
        super().__init__(x, y, z, wavelength)
        self.m = m
        self.n = n
        self.thetax = thetax
        self.thetay = thetay
        self.w0 = w0
        self.zR = np.pi * w0**2 / wavelength
        self.generate()
        self.normalize()


    def generate(self):
        x,y,z = self.x,self.y,self.z
        w0 = self.w0
        w = OpticalFunctions.w(w0,z,self.zR)
        m,n = self.m,self.n
        k = self.k

        H_n = hermite(n)(np.sqrt(2)*x/w)
        H_m = hermite(m)(np.sqrt(2)*y/w)

        amplitude = H_n*np.exp(-x**2/w**2)*H_m*np.exp(-y**2/w**2)
        gouy_phase = self.__gouy__()
        
        if z != 0:
            phase = k * z + k * (x**2+y**2) / (2 * OpticalFunctions.R(z,self.zR)) - gouy_phase
        else:
            phase = self.k * self.z - gouy_phase

        self.E = amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(self.thetax)*self.x) * np.exp(1j*k*np.sin(self.thetay)*self.y)
        
    def intensity(self):
        return np.abs(self.E)**2


    def normalize(self):
        norm = np.sqrt(np.sum(self.intensity()))
        self.E = self.E / norm

    def __gouy__(self):
        return (self.m + self.n + 1) * np.arctan(self.z / self.zR)


class superposition(SpatialMode):
    def __init__(self, modes, x, y, z=0, w0=1e-3, wavelength=632.8e-9):
        super().__init__(x, y, z, w0, wavelength)
        self.modes = modes
        self.generate()
        self.normalize()

    
    def __HermiteMode__(self,**kwargs):
        params = dict(
            w0=1e-3,
            m=0,
            n=0,
            z=0
        )
        params.update(kwargs)

        zR = np.pi * w0**2 / self.wavelength
        
        w0 = params['w0']
        z = params['z']
        m,n = params['m'],params['n']

        w = self.w()

        x,y = self.x,self.y
        k = self.k

        H_n = hermite(n)(np.sqrt(2)*x/w)
        H_m = hermite(m)(np.sqrt(2)*y/w)

        amplitude = H_n*np.exp(-x**2/w**2)*H_m*np.exp(-y**2/w**2)
        gouy_phase = self.__gouy__()
        
        if z != 0:
            phase = k * z + k * (x**2+y**2) / (2 * OpticalFunctions.R(z,self.zR)) - gouy_phase
        else:
            phase = self.k * self.z - gouy_phase

        self.E = amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(self.thetax)*self.x) * np.exp(1j*k*np.sin(self.thetay)*self.y)

    def __LaguerreMode__(self):
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
            phase = k * z + k * r**2 / (2 * OpticalFunctions.R(z,self.zR)) - gouy_phase + l * phi
        else:
            phase = - gouy_phase + l * phi

        self.E = amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(self.thetax)*self.x) * np.exp(1j*k*np.sin(self.thetay)*self.y)
    
    def generate(self):
        modes = self.modes
        for mode in modes:
            if mode[0].lower() == 'hg':
                for i in range(len(mode)-1):
                    amp = mode[1]
                    m = mode[2]
                    n = mode[3]
                    thetax = mode[4]
                    thetay = mode[5]
                

class OpticalFunctions():
    @staticmethod
    def w(w0,z,zR):
        return w0 * np.sqrt(1 + (z / zR)**2)
    @staticmethod
    def R(z,zR):
        return z * (1 + (zR / z)**2)