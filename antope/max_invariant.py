import time
import numpy as np
from .polyhedron import Polyhedron

def max_invariant(A, X, W, Z_inf):
    """Maximal Robust Positively Invariant (MRPI) Set
    Florian Pouthier - PhD student - ICube Strasbourg & GIPSA-lab Grenoble
    Created: January 31st, 2024
    Updated: January 31st, 2024
    Contact: f.pouthier@unistra.fr

    This algorithm provides a computation of the exact maximal robust
    positively invariant (MRPI) set for any linear system.
    
    This code consists in a MATLAB implementation of the algorithm
    detailled in [RF08].

    WARNING: For the moment, this code does not guarantee any convergence

    NOTE [March 11th, 2024]: This is an improved version just
    computing the exterior set. It implies a faster computation, but
    without any guarantee of good result.

    References:
    [RF08] Sasa V. Rakovic, Mirko Fiacchini. Invariant approximations
    of the Maximal Invariant Set or "Encircling the
    Square". Proceedings of the 17th IFAC World Congress, Seoul,
    Korea, (July 6-11, 2008).
    """
    
    # Growing the minimal RPI set approximation to fit X
    # gamma_inf = growth_min(Z_inf,X);
    gamma_inf = 1;
    
    # Initialization of set sequences Y_k and Gamma_k
    Y = gamma_inf * Z_inf
    Gamma_p = X
    Gamma = X.intersect(np.linalg.inv(A)*(Gamma_p-W))
    s = 1
    
    # Optimization loop based on the set-dynamics mapping
    while not Gamma_p.contains(Gamma) or not Gamma.contains(Gamma_p):
        tic = time.time()
        # S1 = Y - W
        # print('S1: ' + str(time.time() - tic))
        # print(str(S1))
        # S1.computeVRep()
        # print(str(S1))
        # A1 = np.linalg.inv(A) * S1
        # print('A1: ' + str(time.time() - tic))
        # print(str(A1))
        # Y = X.intersect(A1)
        # print('Y: ' + str(time.time() - tic))
        Gamma_p = Gamma
        S2 = Gamma_p - W
        # print('S2: ' + str(time.time() - tic))
        A2 = np.linalg.inv(A) * S2
        # print('A2: ' + str(time.time() - tic))
        Gamma = X.intersect(A2)
        # print('Gamma: ' + str(time.time() - tic))
        # Y = X.intersect(np.linalg.inv(A)*(Y-W))
        # Gamma = X.intersect(np.linalg.inv(A)*(Gamma_p-W))
        s = s + 1
    # print("Number of iterations is s=" + str(s))
    return Gamma
