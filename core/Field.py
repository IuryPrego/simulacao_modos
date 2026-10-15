import numpy as np
import matplotlib.pyplot as plt

from core.IntensityModes import laguerre_gauss as Lg
from core.IntensityModes import hermite_gauss as Hg, hermite_gauss_1d as Hg_1d
from core.Polarization_elements import polarizers,scalar_to_vector, filter_polarizer,q_plate as qp
from core.metrics import intensity as I, power as pwr, phase as phi, inner_product as dp
from core.Optical_elements import angular_spectrum as angSpec, tilt, mirror,beam_splitter as bs, lens
from core.masks import pinhole_mask as pinhole, triangular_mask as triangle

# unused code
class OpticalField:
    def __init__(self, x, y, z=0, wavelength=np.array([632.8e-9]), E=None, wavelenght_distribution=None):
        self.x = x
        self.y = y
        self.z = z
        self.dx = float(x[0, 1] - x[0, 0])
        self.dy = float(y[1, 0] - y[0, 0])
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.E = np.zeros([len(wavelength), *x.shape, 2],dtype='complex128')
        if E is not None:
            self.E[:,...] = E


    def Intensity(self,cmap='viridis',vector_field=True,pace=None,scale=None,t=np.pi/2,rel_threshold=0.1,create_canvas=True):
        intensity = np.zeros([len(self.wavelength), *self.x.shape],dtype='float64')
        for i,wl in enumerate(self.wavelength):
            intensity[i] = I(self.E[i],cmap=cmap,vector_field=vector_field,x=self.x,y=self.y,pace=pace,scale=scale,t=t,rel_threshold=rel_threshold,create_canvas=False)[0]
        fig, ax = plt.subplots()
        fig.frameon = False
        ax.axis('equal')
        ax.axis('off')
        intensity_tot = np.sum(intensity, axis=0)
        ax.imshow(intensity_tot,cmap, vmin=0, vmax=max(1e-5,np.max(intensity_tot)),extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],origin='lower')
        return intensity, fig, ax

    def Power(self):
        dx, dy = self.dx,self.dy
        power = np.sum(np.abs(self.E))** 2 * dx * dy
        return power

    def lens(self,focal_length):
        for i,wl in enumerate(self.wavelength):
            self.E[i] = lens(self.E[i],self.x,self.y,f=focal_length,wavelength=wl)

        self.E = lens(self.E,self.x,self.y,f=focal_length,wavelength=self.wavelenght)
        return self

# its wrong, need to be fixed
    def phase(self):
        return np.angle(self.E)

# not that useful
    def normalize(self):
        norm = self.Power()
        if norm != 0:
            self.E /= np.sqrt(norm)
        else:
            raise ValueError('Can`t normalize empty field')

