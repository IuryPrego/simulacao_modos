import numpy as np
from scipy.special import genlaguerre, hermite

def LaguerreGauss(x,y,l=0,p=0,z=0,w0=1e-3,thetax=0,thetay=0,wavelength=632.8e-9,normalize=True):
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    k = 2 * np.pi / wavelength
    zR = np.pi * w0**2 / wavelength
    w = w0 * np.sqrt(1 + (z / zR)**2)
    norm = 0

    L = genlaguerre(p, abs(l))(2*r**2/w**2)

    amplitude = (w0 / w) * ((np.sqrt(2) * r / w)**abs(l)) * L * np.exp(-r**2 / w**2)
    gouy_phase = (2*p + abs(l) + 1) * np.arctan(z / zR)
    tilt = np.exp(1j*k*(thetax)*x) * np.exp(1j*k*(thetay)*y)
    
    if z != 0:
        R = z * (1 + (zR / z)**2)
        phase = k * z + k * r**2 / (2 * R) - gouy_phase + l * phi
    else:
        phase = - gouy_phase + l * phi

    E = amplitude * np.exp(1j * phase) * tilt

    if normalize:
        norm = np.sum(np.abs(E)) ** 2 * dx * dy

    if norm != 0:
        return E/np.sqrt(norm)
    else:
        return E


def Hermite(x,y,m=0,n=0,z=0,w0=1e-3,thetax=0,thetay=0,wavelength=632.8e-9,normalize=True):
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    k = 2 * np.pi / wavelength
    zR = np.pi * w0**2 / wavelength
    w = w0 * np.sqrt(1 + (z / zR)**2)
    norm = 0

    H_n = hermite(n)(np.sqrt(2)*x/w)
    H_m = hermite(m)(np.sqrt(2)*y/w)

    amplitude = H_n*np.exp(-x**2/w**2)*H_m*np.exp(-y**2/w**2)
    gouy_phase = (m + n + 1) * np.arctan(z / zR)
    tilt = np.exp(1j*k*(thetax)*x) * np.exp(1j*k*(thetay)*y)
    
    if z != 0:
        R = z * (1 + (zR / z)**2)
        phase = k * z + k * (x**2+y**2) / (2 * R) - gouy_phase
    else:
        phase = k * z - gouy_phase


    E = amplitude * np.exp(1j * phase) * tilt

    if normalize:
        norm = np.sum(np.abs(E)) ** 2 * dx * dy

    if norm != 0:
        return E/np.sqrt(norm)
    else:
        return E
