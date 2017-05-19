import numpy as np
import numpy.linalg as la

def F(t,x,v):
    # from comparison of Eulerian solvers pg.14
    alpha = 0.1
    k = 0.483
    if (t == 0):
        # single pulse    
        z = alpha*np.cos(k*x)
    else:
        z = 0*x
    return z

def initial_u(x,v):
    z = np.exp(-(v**2)/2)
    z = z/np.sqrt(2*np.pi)
    return z

def central_diff_space(u,x):
    # u is function with no values at x = 0 (periodic)
    # x is spatial grid, including x = 0
    # works for quadratic function (except at boundaries - periodic)
    # consider changing u to u[1:-1,:] as u = 0 for large abs(v)

    dudx = np.roll(u,-1,axis = 1) - np.roll(u,1,axis = 1)
    x_shift = np.roll(x,-1,axis = 1)
    x_shift[:,-1] = x[:,-1] + x[:,1]
    dx = np.diff(x,axis = 1) + np.diff(x_shift,axis = 1)
    dudx = dudx/(dx)
    return dudx[1:-1,:]

def central_diff_vel(u,v):
    #will need to be padded by zeros
    dudv = np.roll(u,-1,axis = 0) - np.roll(u,1,axis = 0)
    dv = np.diff(v,axis = 0)
    dudv = dudv/(dv[:,:-1])
    return dudv[1:-1,:]

 


def Forward_Euler_time(u,dt,dudx,dudv,v,F):
    unew = u*dt - v*dt*dudx - F*dt*dvdx
    return unew
 
