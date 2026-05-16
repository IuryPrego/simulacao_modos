from matplotlib import pyplot as plt
import numpy as np

def inner_product(f, g, dx, dy):
    return np.sum(f * np.conj(g)) * dx * dy

# Plot the intensity and the polarization directions and return the fig,ax to posterior alterations
def intensity(field,cmap='viridis',vector_field=True,pace=None,scale=40,animate=False,t=0,rel_threshold=1e-1):
    field_p = np.copy(field)
    field = np.copy(field)

    if field.ndim == 3:
        field = np.linalg.norm(field,axis=2)
    field = np.abs(field)**2

    fig,ax = plt.subplots()

    fig.frameon = False

    ax.axis('equal')
    ax.axis('off')
    ax.imshow(field,cmap, vmin=0, vmax=max(1e-5,np.max(field)))

    if field_p.ndim == 3 and vector_field:
        x,y = np.meshgrid(np.linspace(0,len(field[0,:]),len(field[0,:])),np.linspace(0,len(field[0,:]),len(field[:,0])))

        field_p[field/field.max() <= rel_threshold] = 0

        Ex = field_p[...,0]
        Ey = field_p[...,1]

        mod = np.sqrt(field)
        mod[mod == 0] = 1

        if animate:
            timephase = np.exp(-1j*t)

            Ex = Ex / mod * timephase
            Ey = Ey / mod * timephase

            u = np.real(Ex)
            v = np.real(Ey)
        else:
            Ex = Ex / mod
            Ey = Ey / mod

            u = np.abs(Ex)
            v = np.abs(Ey)

        if pace == None:
            pace = int(np.min(field.shape)/30)

        ax.quiver(x[::pace, ::pace], y[::pace, ::pace],
                u[::pace, ::pace],v[::pace, ::pace],
                cmap='gray',
                pivot='middle',
                scale=scale)
    
    return fig,ax


# only work in scalar fields
def phase(field):
    return np.angle(field)


# measure of power
def power(x,y,field):
    field = np.copy(field)
    
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    if field.ndim == 3:
        field = np.linalg.norm(field,axis=2)

    power = np.sum(np.abs(field)**2) * dx * dy
    return power
