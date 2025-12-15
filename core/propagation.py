from matplotlib import pyplot as plt
import numpy as np

# propagação usando espectro angular a aproximação paraxial
def angular_spectrum(field, x, y, z, wavelength=632.8e-9):

    nx, ny = field.shape[0], field.shape[1]
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    k = 2*np.pi / wavelength
    kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(ny, d=dy)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')

    field_fourier = np.fft.fft2(field, axes=(0, 1))
    propagator = np.exp(-1j * (kx**2 + ky**2) * z / (2*k))

    if field_fourier.ndim == 3:
        field_fourier *= propagator[..., None]
    else:
        field_fourier *= propagator

    field = np.fft.ifft2(field_fourier, axes=(0, 1))
    
    norm = np.sum(np.abs(field)) ** 2 * dx * dy
    
    if norm != 0:
        return field/np.sqrt(norm)
    else:
        return field

