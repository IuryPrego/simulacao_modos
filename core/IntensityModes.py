from math import factorial

import numpy as np
from scipy.special import genlaguerre, hermite, eval_hermite, gammaln

# creation of intensity modes laguerre and hermite gauss beams
# the functions get x,y and beams parameters to create and return a escalar field

def laguerre_gauss(x,y,l=0,p=0,z=0,w0=1e-3,wavelength=632.8e-9):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    k = 2 * np.pi / wavelength
    zR = np.pi * w0**2 / wavelength
    w = w0 * np.sqrt(1 + (z / zR)**2)
    log_norm = (0.5*(np.log(2) + gammaln(p+1) - np.log(np.pi) - gammaln(p+abs(l)+1))
                - np.log(w))

    L = genlaguerre(p, abs(l))(2*r**2/w**2)

    amplitude = ((np.sqrt(2) * r / w)**abs(l)) * L * np.exp(-r**2 / w**2)
    if z != 0:
        gouy_phase = (2*p + abs(l) + 1) * np.arctan(z / zR)
        R = z * (1 + (zR / z)**2)
        phase = k * z + k * r**2 / (2 * R) - gouy_phase + l * phi
    else:
        phase = l * phi

    E = amplitude * np.exp(1j * phase)

    return E*np.exp(log_norm)

def hermite_gauss(x, y, m=0, n=0, z=0, w0=1e-3, wavelength=632.8e-9):

    k = 2 * np.pi / wavelength
    zR = np.pi * w0**2 / wavelength
    w = w0 * np.sqrt(1 + (z / zR)**2)

    # coordenadas reduzidas
    x1 = x[0, :]
    y1 = y[:, 0]

    a = np.sqrt(2) / w

    # normalização analítica
    log_norm = (- np.log(w)
                + 0.5 * np.log(2/np.pi)
                - 0.5 * ((m+n)*np.log(2) + gammaln(m+1) + gammaln(n+1)))

    # hermites 1D
    Hn = eval_hermite(n, a*y1)
    Hm = eval_hermite(m, a*x1)

    gy = Hn * np.exp(-(a*y1)**2 / 2)
    gx = Hm * np.exp(-(a*x1)**2 / 2)

    amplitude = np.outer(gy, gx)*np.exp(log_norm)

    # fase
    if z != 0:
        gouy = (m + n + 1) * np.arctan(z / zR)
        R = z * (1 + (zR / z)**2)
        phase = k*z + k*(x**2 + y**2)/(2*R) - gouy
    else:
        phase = 0

    return amplitude * np.exp(1j * phase)

def hermite_gauss_1d(x, m, w=1e-3):
    a = np.sqrt(2) / w
    xi = a * x 
    
    log_norm = 0.5*np.log(a) - 0.5 * (m*np.log(2) + gammaln(m+1) + 0.5*np.log(np.pi))
    norm = np.exp(log_norm)
    
    return norm * eval_hermite(m, xi) * np.exp(-xi**2 / 2)