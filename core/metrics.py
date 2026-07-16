from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def inner_product(f, g, dx, dy):
    return np.sum(f * np.conj(g)) * dx * dy

# Plot the intensity and the polarization directions and return the fig,ax to posterior alterations
def intensity(field,cmap='viridis',vector_field=True,pace=None,scale=None,t=None,rel_threshold=1e-1):
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
        Ny,Nx = field.shape

        if field.max() > 0:
            field_p[field/field.max() <= rel_threshold] = 0

        Ex = field_p[...,0]
        Ey = field_p[...,1]

        norm = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
        norm[norm == 0] = 1
        Exn = Ex / norm
        Eyn = Ey / norm

        if pace is None:
            pace = max(int(np.min(field.shape)/20),2)
        if scale is None:
            scale = max(pace/2,.8)
            
        for i in range(0, Ny, pace):
            for j in range(0, Nx, pace):

                if np.linalg.norm(field_p[i, j]) == 0:
                    continue

                ex = Exn[i, j]
                exr = np.abs(ex)
                ey = Eyn[i, j]
                eyr = np.abs(ey)
                eps = np.angle(ey) - np.angle(ex)

                S=exr**2+eyr**2
                P=exr*eyr*np.abs(np.sin(eps))
                sqrt_plus = np.sqrt(S + 2*P)
                sqrt_minus = np.sqrt(S - 2*P)
                den = exr**2 - eyr**2
                den = 0 if np.abs(den) <= 1e-14 else den
                num = 2*exr*eyr*np.cos(eps)
                num = 0 if np.abs(num) <= 1e-14 else num

                w = (sqrt_plus + sqrt_minus)/2
                h = (sqrt_plus - sqrt_minus)/2
                alpha = -np.arctan2(num, den)/2
                # imshow inverts y-axis
                # so alpha = -alpha for correct display

                e = Ellipse(xy=(j, i),
                            width=w*scale,
                            height=h*scale,
                            angle=alpha*180/np.pi,
                            edgecolor='black',
                            facecolor='none')
                ax.add_patch(e)

                a = w/2*scale
                b = h/2*scale
                sng = -1 if np.sin(eps) < 0 else 1
                t_local = t if t is not None else 0 #(0 if sng > 0 else np.pi)
                dt = .01/scale*sng

                xx = j + a*np.cos(t_local-10*dt)*np.cos(alpha) - b*np.sin(t_local-10*dt)*np.sin(alpha)
                yy = i + a*np.cos(t_local-10*dt)*np.sin(alpha) + b*np.sin(t_local-10*dt)*np.cos(alpha)
                ax.plot(xx, yy, alpha=.8, linewidth=.8)

                dxx = (-a*np.sin(t_local-10*dt)*np.cos(alpha) - b*np.cos(t_local-10*dt)*np.sin(alpha))*dt
                dyy = (-a*np.sin(t_local-10*dt)*np.sin(alpha) + b*np.cos(t_local-10*dt)*np.cos(alpha))*dt
               
                ax.annotate(
                    '',
                    xy=(xx+dxx, yy+dyy),
                    xytext=(xx, yy),
                    arrowprops=dict(arrowstyle='->',
                                    color='black',
                                    linewidth=1)
                )

    return fig, ax

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
