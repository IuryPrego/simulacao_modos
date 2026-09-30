from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np

def inner_product(f, g, dx, dy):
    return np.sum(f * np.conj(g)) * dx * dy

# Plot the intensity and the polarization directions and return the fig,ax to posterior alterations
def intensity(field,
              cmap='viridis',
              vector_field=True,
              x=None,y=None,
              pace=None,
              scale=None,
              t=np.pi/2,
              rel_threshold=1e-1):
    field = np.copy(field)
    if x is None:
        x = np.arange(field.shape[1])
    if y is None:
        y = np.arange(field.shape[0])

    if field.ndim == 2:
        intensity = np.abs(field)**2
        fig,ax = plt.subplots()
        
        fig.frameon = False

        ax.axis('equal')
        ax.axis('off')
        ax.imshow(intensity,cmap, vmin=0, vmax=max(1e-5,np.max(intensity)),extent=[x.min(), x.max(), y.min(), y.max()])

    elif field.ndim == 3 and vector_field:
        Ny,Nx = field.shape[:2]
        intensity = np.linalg.norm(field,axis=2)**2
        if intensity.max() != 0:
            field[intensity/intensity.max() <= rel_threshold] = 0

        if pace is None:
            pace = max(int(np.min([Nx,Ny])/15),2)
        if scale is None:
            scale = max(pace/2,1)*.8

        ii = np.arange(0,Ny,pace)
        jj = np.arange(0,Nx,pace)
        ii,jj = np.meshgrid(ii,jj, indexing='ij')

        ii = ii.ravel()
        jj = jj.ravel()
        
        Ex = field[ii,jj,0]
        Ey = field[ii,jj,1]

        norm = np.abs(Ex)**2 + np.abs(Ey)**2 #squared norm in this point

        mask = norm > 0
        ii,jj,Ex,Ey,norm = ii[mask],jj[mask],Ex[mask],Ey[mask],norm[mask]
        
        norm = np.sqrt(norm)
        norm[norm == 0] = 1
        Exn = Ex / norm
        Eyn = Ey / norm
        Ex_real = np.abs(Exn)
        Ey_real = np.abs(Eyn)
        delta_phi = 2*np.pi+np.angle(Eyn) - np.angle(Exn)

        S=Ex_real**2+Ey_real**2
        P=Ex_real*Ey_real*np.abs(np.sin(delta_phi))

        # Clip to zero to prevent small negative values from numerical round-off errors
        # from causing invalid square roots.
        sqrt_plus = np.sqrt(np.clip(S + 2*P,0,None))
        sqrt_minus = np.sqrt(np.clip(S - 2*P,0,None))

        axis_diff = Ex_real**2 - Ey_real**2
        axis_diff[np.abs(axis_diff) <= 1e-14] = 0
        cross_term = 2*Ex_real*Ey_real*np.cos(delta_phi)
        cross_term[np.abs(cross_term) <= 1e-14] = 0
        w = (sqrt_plus + sqrt_minus)/2
        h = (sqrt_plus - sqrt_minus)/2
        alpha = -np.arctan2(cross_term, axis_diff)/2
        # imshow inverts y-axis
        # so alpha = -alpha for correct display

        fig,ax = plt.subplots()
        
        fig.frameon = False

        ax.axis('equal')
        ax.axis('off')
        ax.imshow(intensity,cmap, vmin=0, vmax=max(1e-5,np.max(intensity)),extent=[x.min(), x.max(), y.min(), y.max()])

        ec = EllipseCollection(
            widths=w*scale*2, heights=h*scale*2, angles=alpha*180/np.pi, units='xy',
            offsets=np.column_stack([jj, ii]), offset_transform=ax.transData,
            edgecolor='black', facecolor='none'
            )
        
        ax.add_collection(ec)

        a = w*scale
        b = h*scale
        sin_delta_phi = np.sin(delta_phi)
        sin_delta_phi[np.abs(sin_delta_phi)<=1e-14] = 0
        rotation_direction = np.where(sin_delta_phi < 0,-1,1)
        t_local = t + np.where(rotation_direction > 0,0,np.pi)
        dt = .01
        cos_alpha,sen_alpha = np.cos(alpha),np.sin(alpha)
        cos_t, sen_t = np.cos(t_local-10*dt),np.sin(t_local-10*dt)
        
        xx = jj + a*cos_t*cos_alpha - b*sen_t*sen_alpha
        yy = ii + a*cos_t*sen_alpha + b*sen_t*cos_alpha

        dxx = (-a*sen_t*cos_alpha - b*cos_t*sen_alpha) * dt*rotation_direction
        dyy = (-a*sen_t*sen_alpha + b*cos_t*cos_alpha) * dt*rotation_direction
        
        tangent_norm = np.sqrt(dxx**2 + dyy**2)
        tangent_norm[tangent_norm == 0] = 1
        
        dxx = dxx / tangent_norm * .2*scale
        dyy = dyy / tangent_norm * .2*scale

        ax.quiver(xx-dxx, yy-dyy, dxx, dyy, angles='xy', scale_units='xy', scale=.2,
                units='xy', width=0.08*scale,
                color='black', headwidth=20, headlength=25, headaxislength=20)
    
    elif field.ndim == 3 and not vector_field:
        intensity = np.linalg.norm(field,axis=2)**2
        fig,ax = plt.subplots()
        
        fig.frameon = False

        ax.axis('equal')
        ax.axis('off')
        ax.imshow(intensity,cmap, vmin=0, vmax=max(1e-5,np.max(intensity)),extent=[x.min(), x.max(), y.min(), y.max()])

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
