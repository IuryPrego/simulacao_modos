from matplotlib import pyplot as plt
import numpy as np

def intensity(field,cmap='viridis',t=0,pace=1,scale=None):
    field_p = field[...]
    if field.ndim == 3:
        field = np.linalg.norm(field,axis=2)
    field = np.abs(field)**2

    plt.figure().frameon = False
    plt.axis('equal')
    plt.axis('off')
    plt.imshow(field,cmap, vmin=0, vmax=max(1e-5,np.max(field)))

    if field_p.ndim == 3:
        x,y = np.meshgrid(np.linspace(0,len(field[0,:]),len(field[0,:])),np.linspace(0,len(field[0,:]),len(field[:,0])))
        Ex = field_p[...,0]
        Ey = field_p[...,1]

        mod = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
        mod[mod == 0] = 1

        Ex = Ex / mod * np.exp(-1j*t)
        Ey = Ey / mod * np.exp(-1j*t)

        u = np.real(Ex)
        v = np.real(Ey)

        u[:,0] = 0
        v[:,0] = 0
        u[0,:] = 0
        v[0,:] = 0

        plt.quiver(x[::pace, ::pace], y[::pace, ::pace],
                u[::pace, ::pace],v[::pace, ::pace],
                cmap='gray',
                pivot='middle',
                scale=scale)
    

def phase(field):
    return np.angle(field)

def power(x,y,field):
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    if field.ndim == 3:
        field = np.linalg.norm(field,axis=2)

    power = np.sum(np.abs(field)**2) * dx * dy
    return power
