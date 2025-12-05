import numpy as np


def LinearPolirized(field,theta):
    P = np.zeros_like(field)
    phase_mat = np.array([[np.cos(theta),-np.sin(theta)],
                           np.sin(theta), np.cos(theta)])
    
    

