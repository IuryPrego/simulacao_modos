import numpy as np

def angular_spectrum(x,y,z,field,wavelength):
    nx,ny = field.shape[-1],field.shape[-2]

    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    
    kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(ny, d=dy)
    kx, ky = np.meshgrid(kx, ky)
    k = 2 * np.pi / wavelength

    field_fourier = np.fft.fft2(field)
    propagator = np.exp(-1j*(kx*2+ky**2)*z/(2*k))

    return np.fft.ifft2(field_fourier*propagator)