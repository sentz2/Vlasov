import numpy as np
import matplotlib.pyplot as plt
from FD_Vlasov_supp import *
import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import numpy.linalg as la
import scipy as sci


def FV_step_1(U,V,E,dt,dx,dv):
    # ordered as abo|ve
    U = 0.5*(np.roll(U,-1,axis = 1) + np.roll(U,1,axis = 1) + np.roll(U,-1,axis = 0) + np.roll(U,1,axis = 0)) -\
        U - V*(dt/(2*dx))*(np.roll(U,-1,axis = 1) - np.roll(U,1,axis = 1)) -\
        E*(dt/(2*dv))*(np.roll(U,-1,axis = 0) - np.roll(U,1,axis = 0))
        
    U[0,:] = 0.0
    U[-1,:] = 0.0
    return U

def FV_step_2(U,V,E,dt,dx,dv):
    U = 0.25*(np.roll(U,-1,axis = 1) + np.roll(U,1,axis = 1) + np.roll(U,-1,axis = 0) + np.roll(U,1,axis = 0)) -\
        V*(dt/(2*dx))*(np.roll(U,-1,axis = 1) - np.roll(U,1,axis = 1)) -\
        E*(dt/(2*dv))*(np.roll(U,-1,axis = 0) - np.roll(U,1,axis = 0))
        
    U[0,:] = 0.0
    U[-1,:] = 0.0
    return U

def FV_step_3(U,V,E,dt,dx,dv):
    U_temp = 0.5*(np.roll(U,-1,axis = 1) + np.roll(U,1,axis = 1)) - \
            V*(dt/(2*dx))*(np.roll(U,-1,axis = 1) - np.roll(U,1,axis = 1))
        
    U = 0.5*(np.roll(U_temp,-1,axis = 0) + np.roll(U_temp,1,axis = 0)) -\
        E*(dt/(2*dv))*(np.roll(U_temp,-1,axis = 0) - np.roll(U_temp,1,axis = 0))
        
    U[0,:] = 0.0
    U[-1,:] = 0.0
    return U

Lv = 40     # approximation to "infinity" for velocity space

# see paper for parameters
alpha = 0.01
k = 0.5
L = 2*np.pi/k  # length of spatial domain, which is periodic

# number of cells
nx = 64
nv = 228
dx = L/nx
dv = (2*Lv)/nv

# points are midpoints of each cell
x = np.arange(dx/2,L,dx)
v = np.arange(-Lv + dv/2,Lv,dv)

X,V = np.meshgrid(x,v)  # represent as grid

# "plasma echo"
# k = 0.483
def initial_u(x,v):
    z = np.exp(-(v**2)/2)
    z = z/(np.sqrt(2*np.pi))
    return z

# "nonlinear Landau Damping" - Linear until t < alpha^(-1/2)
# alpha = 0.5
def initial_u2(x,v):
    z = (1+alpha*np.cos(k*x))*np.exp(-(v**2)/2)
    z = z/(np.sqrt(2*np.pi))
    return z

# "two stream instability"
# alpha = 0.01
# k = 0.5
def initial_u3(x,v):
    z = (1 + 5*v**2)*(1 + alpha*((np.cos(2*k*x) + np.cos(3*k*x))/1.2 + np.cos(k*x)))*np.exp(-(v**2)/2)
    z = 2*z/(7*np.sqrt(2*np.pi))
    return z

# todo: make these integrations more accurate by better interpolation/extrapolation
def integrate_udv(u,points):
    v_int = np.linspace(-Lv,Lv,points)
    n = u.shape[1]
    int_u = np.zeros(n,)
    for i in range(n):
        z = sci.interp(v_int,v,U[:,i])
        int_u[i] = integrate.simps(z,v_int)
    return int_u

def integrate_udx(u,points):
    x_int = np.linspace(0,L,points)
    z = sci.interp(x_int,x,u)
    int_u = integrate.simps(z,x_int)
    return int_u

# finite difference solve with homogeneous boundary conditions
# Replace with DG/HDG
def solve_bvp(U,points):
    A = np.diag(2*np.ones(nx-2,)) - np.diag(np.ones(nx-3,),-1) - np.diag(np.ones(nx-3,),1)
    A = A/((x[1] - x[0])**2)   # assumes evenly spaced x
    rho = integrate_udv(U,points) - 1
    phi = np.zeros(nx,)
    phi[1:-1] = la.solve(A,rho[1:-1])
    return phi

def compute_electric_field(U,points):
    dx = x[1] - x[0]
    phi = solve_bvp(U,points)
    E = -np.gradient(phi,dx,edge_order = 2)
    E[-1] = E[0] # might be changed
    E = E.reshape(1,-1)
    E = np.repeat(E,nv,axis = 0)
    return E 


###
# "Fake" electric field
U = initial_u2(X,V)
U[0,:] = 0.0
U[-1,:] = 0.0
dt = 1/50
E = L/2 - X
U = FV_step_3(U,V,E,dt,dx,dv)
steps = 88
for i in range(steps):
    E = L/2 - X
    U = FV_step_3(U,V,E,dt,dx,dv)
    

    
###
# Actually computing electric field
U = initial_u2(X,V)
U[0,:] = 0.0
U[-1,:] = 0.0
dt = 1/50
E = compute_electric_field(U,977)
U = FV_step_3(U,V,E,dt,dx,dv)
steps = 110
for i in range(steps):
    E = compute_electric_field(U,977)
    U = FV_step_3(U,V,E,dt,dx,dv)