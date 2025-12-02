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
        self.zR = np.pi * w0**2 / self._lambda_
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
        self.zR = np.pi * w0**2 / self._lambda_
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


    def __gouy__(self):
        return (self.m + self.n + 1) * np.arctan(self.z / self.zR)


class Superposition(SpatialMode):
    def __init__(self, modes, x, y, z=0, wavelength=632.8e-9):
        super().__init__(x, y, z, wavelength)
        self.modes = modes
        self.generate()
        self.normalize()

    
    def __HermiteMode__(self,**kwargs):
        '''
        Docstring for __HermiteMode__
        
        :param self: Description
        :param kwargs: Beam parameters.
                        - w0 (float): Beam waist
                        - m/n (int): Hermite polynomial index m/n
                        - z (float): Mode's z axis value relative to the choosen point z0 in the creation of the object
                        - thetax/thetay: Incident x/y angle of the beam in the plane
        '''

        
        params = dict(
            w0=1e-3,
            m=0,
            n=0,
            z=0,
            thetax=0,
            thetay=0,
            coef=1
        )
        params.update(kwargs)

        k = self.k
        x,y = self.x,self.y
        dx, dy = self.dx,self.dy
        z = params['z']+self.z
        w0 = params['w0']
        m,n = params['m'],params['n']
        thetax,thetay = params['thetax'],params['thetay']
        coef = params['coef']
        zR = np.pi * w0**2 / self._lambda_
        w = OpticalFunctions.w(w0,z,zR)
        H_n = hermite(n)(np.sqrt(2)*x/w)
        H_m = hermite(m)(np.sqrt(2)*y/w)

        amplitude = H_n*np.exp(-x**2/w**2)*H_m*np.exp(-y**2/w**2)
        gouy_phase = (m + n + 1) * np.arctan(z / zR)
        
        if z != 0:
            phase = k * z + k * (x**2+y**2) / (2 * OpticalFunctions.R(z,zR)) - gouy_phase
        else:
            phase = self.k * self.z - gouy_phase

        E = coef*amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(thetax)*x) * np.exp(1j*k*np.sin(thetay)*y)
        norm = np.sum(np.abs(E)) ** 2 * dx * dy
        
        self.E += E/np.sqrt(norm)


    def __LaguerreMode__(self,**kwargs):
        params = dict(
            w0=1e-3,
            l=0,
            p=0,
            z=0,
            thetax=0,
            thetay=0,
            coef=1
        )
        params.update(kwargs)

        k = self.k
        x,y = self.x,self.y
        dx, dy = self.dx,self.dy
        z = params['z']+self.z
        w0 = params['w0']
        l,p = params['l'],params['p']
        thetax,thetay = params['thetax'],params['thetay']
        coef = params['coef']
        zR = np.pi * w0**2 / self._lambda_
        w = OpticalFunctions.w(w0,z,zR)
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        L = genlaguerre(p, abs(l))(2*r**2/w**2)
        
        amplitude = (w0 / w) * ((np.sqrt(2) * r / w)**abs(l)) * L * np.exp(-r**2 / w**2)
        gouy_phase = (2*p + abs(l) + 1) * np.arctan(z / zR)
        
        if z != 0:
            phase = k * z + k * r**2 / (2 * OpticalFunctions.R(z,zR)) - gouy_phase + l * phi
        else:
            phase = - gouy_phase + l * phi

        E = coef*amplitude * np.exp(1j * phase) * np.exp(1j*k*np.sin(thetax)*x) * np.exp(1j*k*np.sin(thetay)*y)
        norm = np.sum(np.abs(E)) ** 2 * dx * dy
        
        self.E += E/np.sqrt(norm)


    def generate(self):
        modes = self.modes
        for mode in modes:
            if mode['type'].lower() == 'hg':
                self.__HermiteMode__(**mode)
            if mode['type'].lower() == 'lg':
                self.__LaguerreMode__(**mode)
                
        self.normalize()

        
class OpticalFunctions():
    @staticmethod
    def w(w0,z,zR):
        return w0 * np.sqrt(1 + (z / zR)**2)
    @staticmethod
    def R(z,zR):
        return z * (1 + (zR / z)**2)