import numpy as np
from scipy.ndimage import gaussian_filter

#chatgpt
def turbulence_mask(shape, strength=1.0, corr_px=20):
    """
    Máscara de turbulência (fase aleatória correlacionada)
    
    shape     : (ny, nx)
    strength  : força da turbulência (rad)
    corr_px   : comprimento de correlação (pixels)
    """
    phase = np.random.randn(*shape)
    phase = gaussian_filter(phase, corr_px)
    phase *= strength / np.std(phase)

    return np.exp(1j * phase)

import numpy as np

def kolmogorov_phase_screen(nx, ny, dx, dy, r0,
                            L0=np.inf, l0=0.0):
    """
    Tela de fase de turbulência (Kolmogorov / von Kármán)

    nx, ny : número de pontos
    dx, dy : passo espacial (m)
    r0     : parâmetro de Fried (m)
    L0     : outer scale (m)
    l0     : inner scale (m)
    """

    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dy)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')

    kappa = 2*np.pi * np.sqrt(FX**2 + FY**2)
    kappa[0, 0] = np.inf  # evita divergência

    # escalas
    k0 = 0 if np.isinf(L0) else 2*np.pi / L0
    km = np.inf if l0 == 0 else 5.92 / l0

    PSD = 0.023 * r0**(-5/3) * (kappa**2 + k0**2)**(-11/6)
    PSD *= np.exp(-(kappa/km)**2)

    # ruído complexo gaussiano
    cn = (np.random.randn(ny, nx) +
          1j*np.random.randn(ny, nx)) / np.sqrt(2)

    phase_ft = cn * np.sqrt(PSD) * (2*np.pi*np.sqrt(dx*dy))
    phase = np.real(np.fft.ifft2(np.fft.ifftshift(phase_ft)))

    return np.exp(1j * phase), phase

import numpy as np

def turbulence_screen(x, y,
                      r0=8e-4,
                      L0=10.0,
                      l0=5e-4,
                      n_sub=8,
                      seed=None,
                      remove_tilt=False):
    """
    Tela de fase de turbulência com sub-harmônicos (Lane et al.)

    n_sub : número de sub-harmônicos por eixo
    """

    if seed is not None:
        np.random.seed(seed)

    dx = float(x[0,1] - x[0,0])
    dy = float(y[1,0] - y[0,0])
    ny, nx = x.shape

    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dy)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')

    kappa = 2*np.pi*np.sqrt(FX**2 + FY**2)
    kappa[0,0] = 1e-10

    k0 = 2*np.pi/L0 if np.isfinite(L0) else 0
    km = 5.92/l0 if l0 > 0 else np.inf

    PSD = 0.023 * r0**(-5/3) * (kappa**2 + k0**2)**(-11/6)
    if l0 > 0:
        PSD *= np.exp(-(kappa/km)**2)
    PSD[0,0] = 0.0

    noise = (np.random.randn(ny, nx) +
             1j*np.random.randn(ny, nx)) / np.sqrt(2)

    phase_ft = noise * np.sqrt(PSD) * (2*np.pi*np.sqrt(dx*dy))
    phase = np.real(np.fft.ifft2(np.fft.ifftshift(phase_ft)))

    # ---------- SUB-HARMÔNICOS (parte crucial) ----------
    Lx = nx * dx
    Ly = ny * dy

    X = x - np.mean(x)
    Y = y - np.mean(y)

    for p in range(1, n_sub + 1):
        fx_sh = np.array([-1, 0, 1]) / (Lx * p)
        fy_sh = np.array([-1, 0, 1]) / (Ly * p)

        for fx0 in fx_sh:
            for fy0 in fy_sh:
                if fx0 == 0 and fy0 == 0:
                    continue

                k = 2*np.pi*np.sqrt(fx0**2 + fy0**2)
                PSD_sh = 0.023 * r0**(-5/3) * (k**2 + k0**2)**(-11/6)

                amp = np.sqrt(PSD_sh) * (2*np.pi/Lx/Ly)**0.5
                phi = 2*np.pi*(fx0*X + fy0*Y) + 2*np.pi*np.random.rand()

                phase += amp * np.cos(phi)

    if remove_tilt:
        A = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], axis=1)
        c, _, _, _ = np.linalg.lstsq(A, phase.ravel(), rcond=None)
        phase -= (c[0]*X + c[1]*Y + c[2])

    return np.exp(1j*phase), phase
