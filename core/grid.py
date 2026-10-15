import numpy as np
from dataclasses import dataclass, replace
from functools import cached_property

# the transverse grid (x, y) and the wavelength shared by every function of core
# shapes follow numpy: (ny, nx), the same of the fields (ny, nx) or (ny, nx, 2)
# to change something create a new one with replace(s, wavelength=532e-9) or replace(s, x=s.x - x0)
@dataclass(frozen=True, eq=False)
class Setup:
    x: np.ndarray
    y: np.ndarray
    wavelength: float = 632.8e-9

    def __post_init__(self):
        if self.x.ndim != 2 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be 2D arrays with the same shape (use np.meshgrid)")
        # read-only arrays, so changing them outside can't break the setup
        # writeable arrays are copied (they may belong to someone else); read-only ones
        # (from another setup, a replace or a crop) are shared, since nobody can change them
        for name in ('x', 'y'):
            arr = getattr(self, name)
            if arr.flags.writeable or arr.dtype != float:
                arr = np.array(arr, dtype=float)
                arr.flags.writeable = False
            object.__setattr__(self, name, arr)

    # crop the grid like a field: s[i0:i1, j0:j1] goes with field[i0:i1, j0:j1]
    def __getitem__(self, idx):
        return replace(self, x=self.x[idx], y=self.y[idx])

    @property
    def shape(self):
        return self.x.shape

    @property
    def k(self):
        return 2 * np.pi / self.wavelength

    # pixel size, always positive (a reversed grid s[::-1, ::-1] has the same pixels)
    @cached_property
    def dx(self):
        return abs(float(self.x[0, 1] - self.x[0, 0]))

    @cached_property
    def dy(self):
        return abs(float(self.y[1, 0] - self.y[0, 0]))

    @cached_property
    def kxy(self):
        ny, nx = self.shape
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        return np.meshgrid(kx, ky)


# size: physical size (Ly, Lx) in meters; resolution: number of points (ny, nx)
# both accept a scalar for a square grid
def make_setup(size, resolution, wavelength=632.8e-9):
    Ly, Lx = np.broadcast_to(size, 2)
    ny, nx = np.broadcast_to(resolution, 2)
    x, y = np.meshgrid(np.linspace(-Lx/2, Lx/2, nx),
                       np.linspace(-Ly/2, Ly/2, ny))
    # nobody else has these arrays, so they can go read-only to the setup without a copy
    x.flags.writeable = False
    y.flags.writeable = False
    return Setup(x=x, y=y, wavelength=wavelength)


# raises if the field was made in another grid
def check_field(field, s):
    if field.shape[:2] != s.shape:
        raise ValueError(f"field with shape {field.shape[:2]} doesn't match the setup {s.shape}")
