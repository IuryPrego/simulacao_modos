from matplotlib import pyplot as plt
import numpy as np

# propagation using angular espectrum and the paraxial aproximation
def angular_spectrum(field, x, y, z, wavelength=632.8e-9):
    field = np.copy(field)

    nx, ny = field.shape[0], field.shape[1]
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    k = 2*np.pi / wavelength
    kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(ny, d=dy)
    kx, ky = np.meshgrid(kx, ky)

    field_fourier = np.fft.fft2(field, axes=(0, 1))
    propagator = np.exp(-1j * (kx**2 + ky**2) * z / (2*k))

    if field_fourier.ndim == 3:
        field_fourier *= propagator[..., None]
    else:
        field_fourier *= propagator

    field = np.fft.ifft2(field_fourier, axes=(0, 1))
    
    return field

def tilt(field, x, y, thetax=0, thetay=0, wavelength=632.8e-9):
    k = 2 * np.pi / wavelength
    tilt_phase = np.exp(1j * k * thetay * y) * np.exp(1j * k * thetax * x)

    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])

    kmaxx = np.pi / dx
    kmaxy = np.pi / dy

    kx = k * thetax
    ky = k * thetay

    if np.abs(kx) > kmaxx or np.abs(ky) > kmaxy:
        raise ValueError("Tilt muito grande: vai dar aliasing! use no máximo thetax = {} e thetay = {}".format(kmaxx/k, kmaxy/k))
    return np.copy(field) * tilt_phase[...,None] if field.ndim == 3 else np.copy(field) * tilt_phase

def mirror(field, x, y, theta=0, phi=0, rs=-1, rp=-1, wavelength=632.8e-9):
    """
    theta : ângulo do espelho com z (0 = perpendicular ao feixe)
    phi   : orientação azimutal do espelho
    rs, rp: coeficientes de Fresnel (default -1 para espelho metálico ideal)
    """
    # Inverte propagação
    field = np.conj(field)

    # O tilt do espelho equivale a dois tilts (fator 2 = ida e volta)
    thetax = 2 * np.sin(theta) * np.cos(phi)
    thetay = 2 * np.sin(theta) * np.sin(phi)
    field = tilt(field, x, y, thetax=thetax, thetay=thetay, wavelength=wavelength)

    # Rotação de polarização s/p → x/y (só para campos vetoriais)
    if field.ndim == 3:
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[ c,  s],
                      [-s,  c]])
        Jones = np.array([[rp, 0],
                          [0,  rs]])
        M = R.T @ Jones @ R
        field = field @ M.T
    else:
        field = field * rs
    return field