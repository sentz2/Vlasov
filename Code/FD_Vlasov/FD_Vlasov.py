import numpy as np
import matplotlib.pyplot as plt
from FD_Vlasov_supp import *
#import mpl_toolkits.mplot3d.Axes3D as plt3d
import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__
from mpl_toolkits.mplot3d import Axes3D


# based on 1 dimensional example from Comparison of Eulerian Vlasov Solvers

k = 0.483
L = 2*np.pi/k  # length of spatial domain (periodic)
Lv = 20     # approximation to infinity for unbounded velocity domain


nx = 10
nv = 25
xdomain = np.linspace(0,L,nx)
vdomain = np.linspace(-Lv,Lv,nv)
Xdomain, Vdomain = np.meshgrid(xdomain,vdomain)
Uplot = initial_u(Xdomain,Vdomain) # solution for plotting
U = np.delete(Uplot,0,1)  # used for actual computation: the velocity boundaries are "zero"

# do one time step
dudx = central_diff_space(U,X)
dudv = central_diff_vel(U,V)

dt = 0.1
unext = Forward_Euler_time(U,dt,dudx,dudv,V[1:-1,1:],F(0,X[1:-1,1:],V)
