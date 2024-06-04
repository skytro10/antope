import time
import numpy as np
from .polyhedron import Polyhedron

def min_invariant(A, W, eps):
    """Minimal Robust Positively Invariant (mRPI) Set
    Florian Pouthier - PhD student - ICube Strasbourg & GIPSA-lab Grenoble
    Created: January 22nd, 2024
    Updated: January 22nd, 2024
    Contact: f.pouthier@unistra.fr

    This algorithm provides a computation of a robust positively invariant 
    (RPI) approximation of the minimal RPI (mRPI) set from any linear system.
    
    This code consists in a Python implementation of the algorithm detailled
    in [RKKM05] and has been inspired from the below example (written in Julia code):
    https://juliapolyhedra.github.io/Polyhedra.jl/stable/generated/Minimal%20Robust%20Positively%20Invariant%20Set/

    References:
    [RKKM05] Sasa V. Rakovic, Eric C. Kerrigan, Konstantinos I. Kouramas, 
    David Q. Mayne. Invariant approximations of the minimal robust positively 
    Invariant set. IEEE Transactions on Automatic Control 50 (2005): 406-410.
    """

    # Arguments types verification
    if not isinstance(A, np.matrix):
        raise TypeError("'A' argument must be a numpy.matrix for state-space representation") 
    # if not isinstance(W, Polyhedron):
    #     raise TypeError("'W' argument must be a Polyhedron of disturbances")
    if not isinstance(eps, (int, float)):
        raise TypeError("'eps' argument must be a int or float for method precision")

    # Arguments values verification
    global d
    d = W.dim
    if len(A.shape) != 2 or A.shape[0] != W.dim or A.shape[1] != W.dim:
        raise TypeError(f"'A' argument must be a {d}x{d} matrix to agree with W dimension")

    # Computation of an RPI, outer eps-approximation of the mRPI set
    tic = time.time()
    s = 0
    alpha = np.inf
    Mu = np.zeros(d)
    Mb = np.zeros(d)
    Ms = max(max(Mu), max(Mb))
    while alpha > eps/(eps+Ms):
        s = s + 1
        alpha = alpha_0(A, W, s)
        Mu = Mu + sMu(A, W, s) 
        Mb = Mb + sMb(A, W, s) 
        Ms = max(max(Mu), max(Mb))
    Fm = 1/(1-alpha) * Fs(A, W, s)
    toc = time.time() - tic
    return Fm

def alpha_0(A, W, s):
    """Computation of alpha_0(s) 

    Equation (11) in [RKKM05] (polytopic case)
    """
    n = W.H.shape[0] # Number of hyperplanes describing W    
    alpha_list = np.zeros(n)
    for i in range(0, n):
        alpha_list[i] = W.support(np.matmul((A**s).T, W.A[i][:].reshape(d, 1))) / W.b[i]
    return max(alpha_list)

def Fs(A, W, s):
    """Minkowski sum of equation (2) in [RKKM05]"""
    F = W
    A_W = W
    A_W.V
    for i in range(0, s-1):
        # Extend to full state dimension
        # F = Polyhedron('V', [F.V(:,[1,2]) zeros(size(F.V,1),2)]);
        A_W = A * A_W
        F = F + A_W
        # Project the polytope on the plane (omega, Gamma)
        # F = Polyhedron('V', F.V(:,[1,2]));
        # Compute its irredundant V-representation
        F.min_vrep()
        # plot(Polyhedron('V', F.V(:,[1,2])));
    return F
    
def sMu(A, W, s):
    """Summation of positive support functions"""
    Mu = np.zeros(d)
    for i in range(0, d):
        e = np.zeros((d, 1))
        e[i] = 1
        Mu[i] = W.support(np.matmul((A**(s-1)).T, e))
    return Mu
    
def sMb(A, W, s):
    """Summation of negative support functions"""
    Mb = np.zeros(d)
    for i in range(0, d):
        e = np.zeros((d, 1))
        e[i] = 1
        Mb[i] = W.support(-np.matmul((A**(s-1)).T, e))
    return Mb
