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

def mirror(field, x, y, theta=0, phi=0, rs=-1, rp=-1,tilt=False, wavelength=632.8e-9):
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
    if tilt:
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

def BeamSplitter(field1=None,field2=None,theta=np.pi/4, phi_0=0, phi_r=0, phi_t=0, wavelength=632.8e-9):
    if field1 is not None and field2 is not None:
        field1 = np.copy(field1)
        field2 = np.copy(field2)
    elif field1 is not None:
        field1 = np.copy(field1)
        field2 = np.zeros_like(field1)
    elif field2 is not None:
        field2 = np.copy(field2)
        field1 = np.zeros_like(field2)
    else:
        raise ValueError("Pelo menos um dos campos deve ser fornecido")
    
    tau = np.exp(1j * phi_0)*np.array([[np.sin(theta)*np.exp(1j * phi_r), np.cos(theta)*np.exp(-1j * phi_t)],
                                       [np.cos(theta)*np.exp(1j * phi_t), -np.sin(theta)*np.exp(-1j * phi_r)]])
    
    field =np.exp(1j * phi_0)*np.array([np.sin(theta)*np.exp(1j*phi_r) * field1 + np.cos(theta)*np.exp(-1j*phi_t) * field2,
                                        np.cos(theta)*np.exp(1j*phi_t) * field1 - np.sin(theta)*np.exp(-1j*phi_r) * field2])

    return field

def Lens(field, x, y, f, n=1.5, d0 = 0, wavelength=632.8e-9):
    k = 2 * np.pi / wavelength
    lens_phase = np.exp(-1j * k * (x**2 + y**2) / (2*f)) * np.exp(-1j * n * k * d0)
    return np.copy(field) * lens_phase[...,None] if field.ndim == 3 else np.copy(field) * lens_phase