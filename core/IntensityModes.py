import numpy as np
from scipy.special import genlaguerre, hermite, eval_hermite, gammaln
from core.metrics import power
import math

# creation of intensity modes laguerre and hermite gauss beams
# the functions get x,y and beams parameters to create and return a escalar field

def laguerre_gauss(x,y,l=0,p=0,z=0,w0=1e-3,theta_x=0,theta_y=0,wavelength=632.8e-9,normalize=True):
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
    tilt = np.exp(1j*k*(theta_x)*x) * np.exp(1j*k*(theta_y)*y)
    
    if z != 0:
        R = z * (1 + (zR / z)**2)
        phase = k * z + k * r**2 / (2 * R) - gouy_phase + l * phi
    else:
        phase = - gouy_phase + l * phi

    E = amplitude * np.exp(1j * phase) * tilt

    if normalize:
        norm = power(x,y,E)

    if norm != 0:
        return E/np.sqrt(norm)
    else:
        return E


def hermite_gauss(x,y,m=0,n=0,z=0,w0=1e-3,thetax=0,thetay=0,wavelength=632.8e-9,normalize=True):
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
        norm = power(x,y,E)

    if norm != 0:
        return E/np.sqrt(norm)
    else:
        return E


def hermite_gauss_b(x, y, m=0, n=0, z=0, w0=1e-3,
                  thetax=0, thetay=0, wavelength=632.8e-9):

    k = 2 * np.pi / wavelength
    zR = np.pi * w0**2 / wavelength
    w = w0 * np.sqrt(1 + (z / zR)**2)

    # coordenadas reduzidas
    x1 = x[0, :]
    y1 = y[:, 0]

    xi = np.sqrt(2) * x1 / w
    yi = np.sqrt(2) * y1 / w

    # normalização analítica
    log_norm = (
        - np.log(w)
        + 0.5 * np.log(2/np.pi)
        - 0.5 * ((m+n)*np.log(2) + gammaln(m+1) + gammaln(n+1))
        )

    # hermites 1D
    Hn = eval_hermite(n, xi)
    Hm = eval_hermite(m, yi)

    gx = Hn * np.exp(-x1**2 / w**2)
    gy = Hm * np.exp(-y1**2 / w**2)

    amplitude = np.outer(gy, gx)*np.exp(log_norm)

    # fase
    gouy = (m + n + 1) * np.arctan(z / zR)

    if z != 0:
        R = z * (1 + (zR / z)**2)
        phase = k*z + k*(x**2 + y**2)/(2*R) - gouy
    else:
        phase = k*z - gouy

    tilt = np.exp(1j * k * (thetax * x + thetay * y))

    return amplitude * np.exp(1j * phase) * tilt

def hg_1d_n(x, n, w):
    from scipy.special import eval_hermite, gammaln
    
    xi = np.sqrt(2) * x / w
    
    log_norm = -0.5 * (n*np.log(2) + gammaln(n+1) + 0.5*np.log(np.pi))
    norm = np.exp(log_norm)
    
    return norm * eval_hermite(n, xi) * np.exp(-xi**2 / 2)