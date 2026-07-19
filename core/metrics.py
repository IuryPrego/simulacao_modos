from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np

def inner_product(f, g, dx, dy):
    return np.sum(f * np.conj(g)) * dx * dy

# Plot the intensity and the polarization directions and return the fig,ax to posterior alterations
def intensity(field,cmap='viridis',vector_field=True,pace=None,scale=None,t=np.pi/2,rel_threshold=1e-1):
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

        if pace is None:
            pace = max(int(np.min(field.shape)/15),2)
        if scale is None:
            scale = max(pace/2,1)*.8

        ii = np.arange(0,Ny,pace)
        jj = np.arange(0,Nx,pace)
        ii,jj = np.meshgrid(ii,jj, indexing='ij')

        ii = ii.ravel()
        jj = jj.ravel()
        
        Ex = field_p[ii,jj,0]
        Ey = field_p[ii,jj,1]

        norm = np.abs(Ex)**2 + np.abs(Ey)**2 #squared norm in this point

        mask = norm > 0
        ii,jj,Ex,Ey,norm = ii[mask],jj[mask],Ex[mask],Ey[mask],norm[mask]
        
        norm = np.sqrt(norm)
        norm[norm == 0] = 1
        Exn = Ex / norm
        Eyn = Ey / norm
        exr = np.abs(Exn)
        eyr = np.abs(Eyn)
        eps = 2*np.pi+np.angle(Eyn) - np.angle(Exn)

        S=exr**2+eyr**2
        P=exr*eyr*np.abs(np.sin(eps))
        sqrt_plus = np.sqrt(np.clip(S + 2*P,0,None))
        sqrt_minus = np.sqrt(np.clip(S - 2*P,0,None))

        den = exr**2 - eyr**2
        den[np.abs(den) <= 1e-14] = 0
        num = 2*exr*eyr*np.cos(eps)
        num[np.abs(num) <= 1e-14] = 0
        w = (sqrt_plus + sqrt_minus)/2
        h = (sqrt_plus - sqrt_minus)/2
        alpha = -np.arctan2(num, den)/2
        # imshow inverts y-axis
        # so alpha = -alpha for correct display
        ec = EllipseCollection(
            widths=w*scale*2, heights=h*scale*2, angles=alpha*180/np.pi, units='xy',
            offsets=np.column_stack([jj, ii]), offset_transform=ax.transData,
            edgecolor='black', facecolor='none'
            )
        
        ax.add_collection(ec)

        a = w*scale
        b = h*scale
        se = np.sin(eps)
        se[np.abs(se)<=1e-14] = 0
        sgn = np.where(se < 0,-1,1)
        print(sgn)
        t_local = t + np.where(sgn > 0,0,np.pi)
        dt = .01
        ca,sa = np.cos(alpha),np.sin(alpha)
        ct, st = np.cos(t_local-10*dt),np.sin(t_local-10*dt)
        
        xx = jj + a*ct*ca - b*st*sa
        yy = ii + a*ct*sa + b*st*ca

        dxx = (-a*st*ca - b*ct*sa) * dt*sgn
        dyy = (-a*st*sa + b*ct*ca) * dt*sgn
        
        dnorm = np.sqrt(dxx**2 + dyy**2)
        dnorm[dnorm == 0] = 1
        
        dxx = dxx / dnorm * .2*scale
        dyy = dyy / dnorm * .2*scale

        ax.quiver(xx-dxx, yy-dyy, dxx, dyy, angles='xy', scale_units='xy', scale=.2,
                   units='xy', width=0.08*scale,
                   color='black', headwidth=20, headlength=25, headaxislength=20)
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
