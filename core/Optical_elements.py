from matplotlib import pyplot as plt
import numpy as np

from core.grid import check_field

# propagation using angular espectrum and the paraxial aproximation
def angular_spectrum(field, s, z):
    check_field(field, s)
    k = s.k
    kx, ky = s.kxy

    field_fourier = np.fft.fft2(field, axes=(0, 1))
    propagator = np.exp(-1j * (kx**2 + ky**2) * z / (2*k))* np.exp(1j * k * z)

    if field_fourier.ndim == 3:
        field_fourier *= propagator[..., None]
    else:
        field_fourier *= propagator

    field = np.fft.ifft2(field_fourier, axes=(0, 1))
    
    return field

def tilt(field, s, thetax=0, thetay=0, safe=True):
    check_field(field, s)
    k = s.k
    tilt_phase = np.exp(1j * k * thetay * s.y) * np.exp(1j * k * thetax * s.x)
    if safe:
        kmaxx = np.pi / s.dx
        kmaxy = np.pi / s.dy

        kx = k * thetax
        ky = k * thetay

        if np.abs(kx) > kmaxx or np.abs(ky) > kmaxy:
            raise ValueError("Tilt muito grande: vai dar aliasing! use no máximo thetax = {} e thetay = {}".format(kmaxx/k, kmaxy/k))
    return np.copy(field) * tilt_phase[...,None] if field.ndim == 3 else np.copy(field) * tilt_phase

def mirror(field, theta=0, phi=0, rs=-1, rp=-1,do_tilt=False,safe = False, s=None):
    """
    theta : ângulo do espelho com z (0 = perpendicular ao feixe)
    phi   : orientação azimutal do espelho
    rs, rp: coeficientes de Fresnel (default -1 para espelho metálico ideal)
    s     : setup, só é necessário com do_tilt=True
    """
    # O tilt do espelho equivale a dois tilts (fator 2 = ida e volta)
    if do_tilt:
        if s is None:
            raise ValueError("do_tilt=True needs the setup: mirror(..., s=s)")
        thetax = 2 * np.sin(theta) * np.cos(phi)
        thetay = 2 * np.sin(theta) * np.sin(phi)
        field = tilt(field, s, thetax=thetax, thetay=thetay, safe=safe)

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

def beam_splitter(field1,field2=None,theta=np.pi/4, phi_0=np.pi/2, phi_r=-np.pi/2, phi_t=0):

    if field2 is not None:
        field2 = np.copy(field2)
    else:
        field2 = np.zeros_like(field1)
    
    return np.exp(1j * phi_0) * (np.sin(theta)*np.exp(1j*phi_r) * field1 + np.cos(theta)*np.exp(-1j*phi_t) * field2), np.exp(1j * phi_0)*(np.cos(theta)*np.exp(1j*phi_t) * field1 - np.sin(theta)*np.exp(-1j*phi_r) * field2)

def lens(field, s, f, n=1.5, d0 = 0):
    check_field(field, s)
    field = np.copy(field)
    k = s.k
    lens_phase = np.exp(-1j * k * (s.x**2 + s.y**2) / (2*f)) * np.exp(-1j * n * k * d0)
    return field * lens_phase[...,None] if field.ndim == 3 else field * lens_phase