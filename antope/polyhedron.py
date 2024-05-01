import numpy as np
import cdd
import cProfile
import time
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

class Polyhedron:
  """Polyhedron object with properties

  P = Polyhedron().                              Create empty Polyhedron
  P = Polyhedron(V),    V = matrix.              Create V-rep Polyhedron
  P = Polyhedron(A, b), A = matrix, b = vector.  Create H-rep Polyhedron
  P = Polyhedron('H'=H, 'V'=V), V, H = matrices. Create H-rep and V-rep Polyhedron

  Mainly inspired by the implementation of the MPT 3.0:
  M. Herceg, M. Kvasnica, C.N. Jones, and M. Morari. Multi-Parametric
  Toolbox 3.0. In Proc. of the European Control Conference, pages
  502–510, Zurich, Switzerland, July 17–19 2013.
  https://www.mpt3.org/

  N.B.: The current class does not implement the support for rays
  """

  def __init__(self, *args, **kwargs):
    self.irredundantHRep = False # True if the H-representation is irredundant
    self.irredundantVRep = False # True if the V-representation is irredundant

    self._H = np.empty((0, 1)) # Inequality description { x | H*[x; -1] <= 0 }
    self._V = np.empty((0, 0)) # Vertices of the polyhedron

    if any(kw not in ('V', 'H', 'A', 'b') for kw in kwargs):
      raise TypeError('Keyword arguments can only be among the following: V, H, or A and b')

    if len(args) == 1 or any(kw in kwargs for kw in ('V', 'R')):
      if isinstance(args[0], (np.ndarray, list, tuple)):
        if 'H' in kwargs: # Polyhedron defined by H-rep
          self._set_H(kwargs.get('H'))
        else: # Polyhedron defined by V-rep
          self._set_V(args[0])
      else:
        raise TypeError('Argument must be a numpy.ndarray (list of vertices) for V-rep')

    elif len(args) == 2:
      # Polyhedron defined by H-rep with provided A and b matrices
      A, b = args
      if isinstance(A, (np.ndarray, list, tuple)) and isinstance(b, (np.ndarray, list, tuple)):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)# [:, np.newaxis]
        if len(A.shape) != 2 or len(b.shape) != 2:
          raise ValueError('A and b matrices must be 2-dimensional')
        if A.shape[0] != b.shape[0]:
          e = ['A and b matrices must define the same number of hyperplanes ']
          e += [f'({A.shape[0]} != {b.shape[0]})']
          raise ValueError(''.join(e))
        self._set_H(np.hstack((A, b)))
      else:
        raise TypeError('Arguments must be numpy.ndarrays (A and b matrices) for H-rep')

    elif len(args) == 3:
      # Polyhedron defined by V-rep and H-rep (V, A and b matrices)
      V, A, b = args # TODO

    elif len(args) > 3:
      raise TypeError(f'Polyhedron expected at most 3 arguments, got {len(args)}')

    # elif len(args) > 2:
    #   raise ValueError('Cannot specify V in addition to A and b')

  #   if V_or_R_passed and A_and_b_passed:
  #     raise ValueError('Cannot specify V in addition to A and b')

  #   if (V_or_R_passed or A_and_b_passed) and lb_or_ub_passed:
  #     raise ValueError('Cannot specify bounds in addition to V, R, A, or b')

  #   if ('A' in kwargs) ^ ('b' in kwargs): # XOR
  #     raise ValueError('Cannot pass just one of A and b as keywords')

    # Parse V if passed as the only positional argument or as a keyword
    # if V_or_R_passed:
    #   V = kwargs.get('V')  # None if not
    #   if len(args) == 1:  # V is the only positional argument
    #     # Prevent passing vertex lists as first positional argument and as
    #     # keyword, as in P = Polytope(V_list1, V=V_list2):
    #     if V is not None:
    #       raise ValueError('V cannot be passed as the first argument and as a '
    #                        'keyword')
    #     V = args[0]  # The first positional argument is V
      # Parse R if passed as keyword

  #   # Parse A and b if passed as first two positional arguments or as keywords
  #   if A_and_b_passed:
  #     A = kwargs.get('A')  # None if not
  #     b = kwargs.get('b')  # None if not
  #     # Prevent passing A or b in both args and kwargs:
  #     if len(args) == 2:  # A and b passed in args
  #       if A is not None or b is not None:  # A or b passed in kwargs
  #         raise ValueError(('A (or b) cannot be passed as the first (or second)'
  #                           ' argument and as a keyword'))
  #       A, b = args[:2]  # The first two positional arguments are A and b
  #     self._set_Ab(A, b)

  #   if lb_or_ub_passed:
  #     # Parse lower and upper bounds. Defaults to [], rather than None,
  #     # if key is not in kwargs (cleaner below).
  #     lb = np.atleast_1d(np.squeeze(kwargs.get('lb', [])))
  #     ub = np.atleast_1d(np.squeeze(kwargs.get('ub', [])))
  #     if (lb > ub).any():
  #       raise ValueError('No lower bound can be greater than an upper bound')
  #     self._set_Ab_from_bounds(lb, ub)  # sets A, b, n, and in_H_rep (to True)

  # To enable linear mapping (a numpy ndarray M multiplies a polytope P: M * P)
  # with the * operator, set __array_ufunc__ = None so that the result is not
  # M.__mul__(P), which is an ndarray of scalar multiples of P: {m_ij * P}. With
  # __array_ufunc__ set to None here, the result of M * P is P.__rmul__(M),
  # which calls P.linear_map(M). See
  # https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
  # An alternative is to set __array_priority__ = 1000 or some other high value.
  __array_ufunc__ = None

  @property
  def A(self):
    if not self.hasHRep and self.hasVRep:
      self.computeHRep()
    return self._H[:, 0:-1].reshape(self.H.shape[0], self.dim)

  @property
  def b(self):
    if not self.hasHRep and self.hasVRep:
      self.computeHRep()
    return self._H[:, -1].reshape(self.H.shape[0], 1)

  @property
  def dim(self):
    if self.hasVRep:
      return self._V.shape[1]
    elif self.hasHRep:
      return self._H.shape[1]-1
    else:
      return 0

  @property
  def H(self):
    if not self.hasHRep and self.hasVRep:
      self.computeHRep()
    return self._H

  @H.setter
  def H(self, H):
    return self._set_H(H)

  def _set_H(self, H):
    self._H = np.array(H, dtype=float)

  @property
  def hasHRep(self):
    return True if self._H.shape[1] > 1 else False

  @property
  def hasVRep(self):
    return True if self._V.shape[1] > 0 else False

  # def _set_Ab(self, A, b):
  #   A = np.array(A, ndmin=2)  # ndmin=2 prevents shape (n,) when m = 1
  #   b = np.squeeze(b)[:, np.newaxis]  # overlaps with _set_b(b)
  #   m, n = A.shape
  #   if b.shape[0] != m:
  #     raise ValueError(f'A has {m} rows; b has {b.shape[0]} rows!')
  #   # For rows with b = +- inf: set a_i = 0' and b_i = +- 1 (indicates which
  #   # direction the constraint was unbounded -- not important)
  #   inf_rows = np.squeeze(np.isinf(b))  # True also for -np.inf
  #   if inf_rows.any():
  #     A[inf_rows, :] = 0
  #     b[inf_rows] = 1 * np.sign(b[inf_rows])
  #   self._set_A(A)
  #   self._set_b(b)
  #   self.n = n
  #   if A.shape[0] and b.shape[0]:  # in_H_rep if A and b are not empty arrays
  #     self.in_H_rep = True

  # def _set_Ab_from_bounds(self, lb, ub):
  #   A_bound = []
  #   b_bound = []
  #   n = np.max((lb.size, ub.size))  # 0 for bound not specified
  #   # Test size of lb instead of "if lb" since the bool value of
  #   # np.array(None) is False whereas np.array([]) is ambiguous. Since
  #   # np.array(None).size is 1 and np.array([]).size is 0 (default, set
  #   # above), lb.size > 0 implies lb was a kwarg.
  #   if lb.size > 0:
  #     if not lb.size == n:
  #       raise ValueError((f'Dimension of lower bound lb is {lb.size}; '
  #                         f'should be {n}'))
  #     A_bound.extend(-np.eye(n))
  #     b_bound.extend(-lb)
  #   if ub.size > 0:
  #     if not ub.size == n:
  #       raise ValueError((f'Dimension of upper bound ub is f{ub.size}; '
  #                         f'should be {n}'))
  #     A_bound.extend(np.eye(n))
  #     b_bound.extend(ub)
  #     self._set_Ab(A_bound, b_bound)  # sets n and in_H_rep to True

  @property
  def V(self):
    if not self.hasVRep and self.hasHRep:
      self.computeVRep()
    return self._V

  @V.setter
  def V(self, V):
    return self._set_V(V)

  def _set_V(self, V):
    self._V = np.array(V, dtype=float)

  # @property
  # def centroid(self):
  #   return np.sum(self.V, axis=0) / self.nV

  # def V_sorted(self):
  #   # Sort vertices (increasing angle: the point (x1, x2) = (1, 0) has angle 0).
  #   # np.arctan2(y, x) returns angles in the range [-pi, pi], so vertices are
  #   # sorted clockwise from 9:00 (angle pi). Note that the first argument is y.
  #   # Mainly for plotting and not implemented for n != 2.
  #   if self.n != 2:
  #     raise  ValueError('V_sorted() not implemented for n != 2')
  #   c = self.centroid
  #   order = np.argsort(np.arctan2(self.V[:, 1] - c[1], self.V[:, 0] - c[0]))
  #   return self.V[order, :]

  def __add__(self, other):
    """Compute the Minkowski sum with another polyhedron or translation by vector
    -----------------------------------------------------------------------------
    Case 1: x is a vector of length P.Dim

    Px = P + x

    Computes the Minkowski sum:

    P+x = {x+y | y in P}

    Inputs:
    P: polyhedron in the n-dimensional space
    x: point in the n-dimensional space as a column vector

    Returns:
    Px - Minkowski sum of this polyhedron P and x
    """
    if isinstance(other, Polyhedron):
      # Polyhedra must be the same dimension for Minkowski sum
      if self.dim != other.dim:
        raise ValueError(f'Polyhedra must be the same dimension ({self.dim} != {other.dim})')
      else:
        return self.minkowskiSum(other)
    elif isinstance(other, (np.ndarray, list, tuple)):
      return self.translation(other)
    else:
      raise TypeError('Right operand must be a numpy.ndarray (vector) or a Polyhedron')

  def __bool__(self):
    return self.hasHRep or self.hasVRep # Return True if the polytope is not empty

  def __mul__(self, other):
    if isinstance(other, np.ndarray):
      return self.invAffineMap(other)
    elif isinstance(other, (int, float)):
      return self.scaling(other)
    elif isinstance(other, Polyhedron):
      return self.cartesianProduct(other)
    else:
      raise TypeError('Right operand must be a numpy.ndarray (list), int, float, or a Polyhedron')

  # def __neg__(self):
  #   neg_P = Polytope()
  #   if self.in_V_rep:
  #     neg_P.V = -self.V
  #   if self.in_H_rep:
  #     neg_P._set_Ab(-self.A, self.b)
  #   return neg_P

  def __radd__(self, other):
    if isinstance(other, (np.ndarray, list, tuple)):
      return self.translation(other)
    else:
      raise TypeError('Left operand must be a numpy.ndarray, list or tuple')

  def __repr__(self):
    if self.dim == 0:
      r = ['Empty polyhedron in R^0']
    else:
      r = [f'Polyhedron in R^{self.dim} with representations:\n']
      r += [f'\tH-rep ']
      if self.hasHRep:
        r += ['(irredundant)' if self.irredundantHRep else '(redundant)  ']
        r += [f' : Inequalities {self._H.shape[0]}\n']
      else:
        r += ['              : Unknown (call computeHRep() to compute)\n']
      r += [f'\tV-rep ']
      if self.hasVRep:
        r += ['(irredundant)' if self.irredundantVRep else '(redundant)  ']
        r += [f' : Vertices {self._V.shape[0]}\n']
      else:
        r += ['              : Unknown (call computeVRep() to compute)\n']
      r += ['Functions : none']
    return ''.join(r)

  def __rmul__(self, other):
    """Compute the Polyhedron multiplication by a matrix or scalar as left operand

    Case 1: numpy.ndarray (matrix) as left operand: call the affineMap() method
    Case 2: int or float (scalar) as left operand: call the scaling() method
    """
    if isinstance(other, np.ndarray):
      return self.affineMap(other)
    elif isinstance(other, (int, float)):
      return self.scaling(other)
    else:
      raise TypeError('Left operand must be a numpy.ndarray, int or float')

  def __rsub__(self, other):
    """Polyhedron substraction only defined for Polyhedron as left operand

    In this case, __sub__ is called, therefore __rsub__ is not defined
    """
    raise TypeError('Left operand must be a Polyhedron')

  def __str__(self):
    r = ['    Polyhedron with properties:\n\n']
    r += [f'        irredundantHRep: {self.irredundantHRep}\n']
    r += [f'        irredundantVRep: {self.irredundantVRep}\n']
    r += [f'                hasHRep: {self.hasHRep}\n']
    r += [f'                hasVRep: {self.hasVRep}\n']
    r += [f'                      A: {self._H[:, 0:-1].shape} {self._H.dtype}\n']
    r += [f'                      b: {self._H[:, -1].shape} {self._H.dtype}\n']
    r += [f'                      H: {self._H.shape} {self._H.dtype}\n']
    r += [f'                      V: {self._V.shape} {self._V.dtype}\n']
    r += [f'                    Dim: {self.dim}\n']
    r += [f'                   Data: []\n']
    return ''.join(r)

  def __sub__(self, other):
    """Compute the Minkowski difference with polyhedron or translation by vector
    ----------------------------------------------------------------------------
    Case 1: x is a vector of length P.Dim

    Px = P + x

    Computes the Minkowski sum:

    P+x = {x+y | y in P}

    Inputs:
    P: polyhedron in the n-dimensional space
    x: point in the n-dimensional space as a column vector

    Returns:
    Px - Minkowski sum of this polyhedron P and x
    """
    if isinstance(other, Polyhedron):
      # Polyhedra must be the same dimension for Pontryagin difference
      if self.dim != other.dim:
        raise ValueError(f'Polyhedra must be the same dimension ({self.dim} != {other.dim})')
      else:
        return self.pontryaginDiff(other)
    elif isinstance(other, (np.ndarray, list, tuple)):
      return self.translation(other, substract=True)
    else:
      raise TypeError('Right operand must be a numpy.ndarray (vector) or a Polyhedron')

  # def multiply(self, other, inverse=False):
  #   # scale: s * P or P * s with s a scalar and P a polytope
  #   # linear map: M * P with M a matrix
  #   # inverse linear map: P * M with M a matrix
  #   if isinstance(other, Polytope):
  #     raise ValueError('Product of two polytopes not implemented')
  #   # TODO: now assuming a numeric type that can be squeezed -- fix
  #   # other can be a scalar (ndim=0) or a matrix (ndim=2)
  #   factor = np.squeeze(other)
  #   if factor.ndim == 0:
  #     return scale(self, other)
  #   elif factor.ndim == 2:
  #     if inverse:
  #       raise ValueError('Inverse linear map P * M not implemeted')
  #     else:
  #       return linear_map(other, self)
  #   else:
  #     raise ValueError('Mulitplication with numeric type other than scalar and '
  #                      'matrix not implemented')

  def affineMap(self, matrix):
    """Compute the affine map of the polyhedron with a matrix of size n x dim

    If n  < dim then this is projection
    If n  > dim then this is a lifting
    If n == dim then this is rotation/skew

    The output is a new polyhedron based on the affine map matrix
    """
    if not isinstance(matrix, np.ndarray):
      raise TypeError(f'affineMap() argument must be a numpy.ndarray, not {type(matrix)}')
    if matrix.shape[1] != self.dim:
      raise ValueError(f'The matrix defining the affine map must have {self.dim} columns.')
    if len(matrix.shape) != 2:
      raise ValueError(f'The matrix must be a 2-dimensional numpy.ndarray.')
    if self.dim == 0: # Empty set case
      return Polyhedron()
    # TODO: Special case: zero map?
    if self.hasVRep: # and self.hasHRep:
      # self.computeVRep()
      return Polyhedron(np.matmul(self._V, matrix.T))
    else:
      return Polyhedron(np.matmul(self.A, np.linalg.inv(matrix)), self.b)

  def cartesianProduct(self, poly):
    """Compute the cartesian product of the polyhedron with another polyhedron

    The output is (for now) just the convex hull of the result
    """
    self.minHRep
    poly.minHRep
    return Polyhedron(np.block([[self.A, np.zeros((self.A.shape[0], self.dim))],
                                [np.zeros((poly.A.shape[0], poly.dim)), poly.A]]),
                      np.vstack((self.b, poly.b)))

  def computeHRep(self):
    """V-rep to H-rep conversion with possibly redundant H-rep output

    Based on ConvexHull from scipy.spatial
    """
    if self.hasHRep or not self.hasVRep: # Nothing to do
      return
    # elif self.isFullSpace:
    # R^n case
    # TODO
    self.minVRep() # Working with minimal V-rep for better computation
    hull = ConvexHull(self.V)
    simplices = self.V[hull.simplices]
    As = np.zeros((simplices.shape[0], self.dim))
    bs = np.ones((simplices.shape[0], 1))
    for i in range(0, simplices.shape[0]):
      W = np.hstack((simplices[i,:], -np.ones((self.dim, 1))))
      U, S, V = np.linalg.svd(W.T) # Null-space computation
      A = U[0:self.dim,-1]
      b = U[-1,-1]
      As[i,:] = A
      bs[i] = b
    self._set_H(np.hstack((As, bs)))

  def computeVRep(self):
    """H-rep to V-rep conversion with possibly redundant V-rep output

    Based on get_generators from cdd
    """
    if self.hasVRep or not self.hasHRep: # Nothing to do
      return
    self.minHRep() # Working with minimal H-rep for better computation
    # print("minHrep" + str(self))
    # Vertex enumeration from halfspace representation using cdd
    H = np.hstack((self.b, -self.A))  # [b, -A]
    H = cdd.Matrix(H, number_type='float')
    H.rep_type = cdd.RepType.INEQUALITY
    H_P = cdd.Polyhedron(H)
    P_tV = H_P.get_generators()  # type(P_tV):  <class 'cdd.Matrix'>
    tV = np.array(P_tV[:])
    # print(tV)
    if tV.any(): # tV == [] if the Polytope is empty
      V_rows = tV[:, 0] == 1  # bool array of which rows contain vertices
      R_rows = tV[:, 0] == 0  # and which contain rays (~ V_rows)
      V = tV[V_rows, 1:]  # array of vertices (one per row)
      R = tV[R_rows, 1:]  # and of rays
      if R_rows.any():
        raise ValueError('Support for rays not implemented')
    else:
      V = np.empty((0, self.dim))
    self._set_V(V)

    # TODO: shift the polytope to the center? (centroid? Chebyshev center?)
    # cdd uses the halfspace representation [b, -A] | b - Ax >= 0

    # From the get_generators() documentation: For a polyhedron described as
    #   P = conv(v_1, ..., v_n) + nonneg(r_1, ..., r_s),
    # the V-representation matrix is [t V] where t is the column vector with
    # n ones followed by s zeroes, and V is the stacked matrix of n vertex
    # row vectors on top of s ray row vectors.

  def contains(self, other):
    """Test whether a (list of) point(s) or a Polyhedron is contained in this Polyhedron"""
    if isinstance(other, Polyhedron):
      if other.dim != self.dim:
        raise ValueError(f'Polyhedra must be the same dimension ({self.dim} != {other.dim})')
      return all(self.contains(other.V.T))
    elif isinstance(other, (np.ndarray, list, tuple)):
      x = np.array(np.squeeze(other), dtype=float)
      if x.shape[0] != self.dim:
        raise ValueError('Polyhedron and list of vertices must be the same dimension'
                         f'({self.dim} != {x.shape[0]})')
      test = np.matmul(self.A, x) - self.b
      is_contained = np.logical_or(test < 0, np.isclose(test, 0))
      return np.all(is_contained, axis=0)
    else:
      raise TypeError('contains() argument must be a list of vertices or a Polyhedron,'
                      f'not {type(poly)}')

  def intersect(self, poly):
    """Compute the intersection of the given polyhedron with this one."""
    if isinstance(poly, Polyhedron):
      if poly.dim != self.dim:
        raise ValueError(f'Polyhedra must be the same dimension ({self.dim} != {poly.dim})')
      return Polyhedron(np.vstack((self.A, poly.A)), np.vstack((self.b, poly.b)))
    else:
      raise TypeError(f'intersect() argument must be a Polyhedron, not {type(poly)}')

  def invAffineMap(self, matrix):
    """Compute the inverse affine map of the Polyhedron with a matrix of size n x dim

    TODO: Write this function
    """
    # if not isinstance(matrix, np.ndarray):
    #   raise TypeError(f'affineMap() argument must be a numpy.ndarray, not {type(matrix)}')
    # if matrix.shape[1] != self.dim:
    #   raise ValueError(f'The matrix defining the affine map must have {self.dim} columns.')
    # if len(matrix.shape) != 2:
    #   raise ValueError(f'The matrix must be a 2-dimensional numpy.ndarray.')
    # if self.dim == 0: # Empty set case
    #   return Polyhedron()
    # # TODO: Special case: zero map?
    # if self.hasVRep:
    #   return Polyhedron(np.matmul(self._V, matrix.T))
    # else:
    pass # TODO: Manage H-rep case

  def minHRep(self):
    if not self.hasHRep:
      self.computeHRep # Convert to H-rep if necessary
    if self.irredundantHRep:
      return
    # hull = ConvexHull(self.V)
    # self._set_V(self.V[hull.vertices])
    # self.irredundantVRep = True

  def minVRep(self):
    if not self.hasVRep:
      self.computeVRep # Convert to V-rep if necessary
    if self.irredundantVRep:
      return # Nothing to do
    hull = ConvexHull(self.V)
    self.V = self.V[hull.vertices]
    self.irredundantVRep = True

  def pontryaginDiff(self, poly):
    """Compute the minkowski difference of another Polyhedron with this Polyhedron

    # TODO
    # In MPT, computation only based on H-rep
    # See if efficient computation exists for V-rep
    """
    if not isinstance(poly, Polyhedron):
      raise TypeError(f'pontryaginDiff() argument must be a Polyhedron, not {type(poly)}')
    if self == poly:
      # Warning: define == operator
      # Warning: define empty polyhedron in the same dimension
      return Polyhedron() # Special case P == S
    self.minHRep
    poly.minHRep
    tb = np.full((self.A.shape[0], 1), np.nan)
    for i in range(0, self.A.shape[0]):
      # cProfile.run('sup = poly.support(self.A[i])')
      tb[i] = self.b[i] - poly.support(self.A[i].T)
    return Polyhedron(self.A, tb)

  def minkowskiSum(self, poly):
    if self.hasVRep and poly.hasVRep:
      Va = self.V
      Vb = poly.V
      if Va.shape[0] == 0:
        Vs = Va
      elif Vb.shape[0] == 0:
        Vs = Vb
      else:
        Vs = np.zeros((Va.shape[0]*Vb.shape[0], poly.dim))
        for i in range(0, Vb.shape[0]):
          Vs[i*Va.shape[0]:(i+1)*Va.shape[0],:]=Va+Vb[i,:]
      return Polyhedron(Vs)

  # def plot(self, ax, **kwargs):
  #   # Plot Polytope. Add separate patches for the fill and the edge, so that
  #   # the fill is below the gridlines (at zorder 0.4) and the edge edge is
  #   # above (at zorder 2, same as regular plot). Gridlines are at zorder 1.5
  #   # by default and at  (1 is default), which is below gridlines, which are
  #   # at zorder 1.5 by default 0.5 if setting ax.set_axisbelow(True),
  #   # so plotting the fill at 0.4 ensures the fill is always below the grid.
  #   h_patch = [] # handle, return as tuple
  #   V_sorted = self.V_sorted()
  #   # Check for edgecolor. Default is (0, 0, 0, 0), with the fourth 0 being
  #   # alpha (0 is fully transparent). Passingedgecolor='r', e.g., later
  #   # translates to (1.0, 0.0, 0.0, 1).
  #   edgecolor_default = (0, 0, 0, 0)
  #   edgecolor = kwargs.pop('edgecolor', edgecolor_default)
  #   # Handle specific cases: edge transparent or set explicitly to None
  #   if ((type(edgecolor) is tuple and len(edgecolor) == 4 and edgecolor[3] == 0)
  #       or edgecolor is None):
  #     edgecolor = edgecolor_default
  #   # Added keyword for edge transparency: edgealpha (default: 1.0)
  #   edgealpha_default = 1.0
  #   edgealpha = kwargs.pop('edgealpha', edgealpha_default)
  #   # Check for fill and facecolor. fill is default True. The Polygon
  #   # documentation lists facecolor=None as valid but it results in blue
  #   # filling (preserving this behavior).
  #   fill = kwargs.pop('fill', True)
  #   facecolor = kwargs.pop('facecolor', None)
  #   # test for any non-empty string and rbg tuple, and handle black as rbg
  #   if any(edgecolor) or edgecolor == (0, 0, 0):
  #     # Plot edge:
  #     temp_dict = {**kwargs,
  #                  **{'fill': False, 'edgecolor': edgecolor, 'zorder': 2,
  #                     'alpha': edgealpha}}
  #     h_patch.append(ax.add_patch(Polygon(V_sorted, **temp_dict)))
  #   if fill or facecolor:
  #     # Plot fill:
  #     temp_dict = {**kwargs,
  #                  **{'edgecolor': None, 'facecolor': facecolor, 'zorder': 0.4}}
  #     h_patch.append(ax.add_patch(Polygon(V_sorted, **temp_dict)))
  #   return tuple(h_patch)  # handle(s) to the patch(es)

  # def plot_basic(self, ax, **kwargs):
  #   h_patch = ax.add_patch(Polygon(self.V_sorted(), **kwargs))
  #   return h_patch # handle to the patch

  def scaling(self, scalar):
    """Compute the polyhedron scaling by some scalar"""
    if not isinstance(scalar, (int, float)):
      raise TypeError(f'scaling() argument must be an int or float, not {type(scalar)}')
    if self.hasVRep:
      return Polyhedron(scalar * self.V)
    if self.hasHRep:
      # pass # TODO: Manage H-rep case
      return Polyhedron(scalar * self.V)
      # np.hstack((self.A, scalar*self.b))

  def sortVertices(self):
    if self.hasHRep and not self.hasVRep:
      self.computeVRep()
    angles = np.arctan2(self._V[:,1], self._V[:,0])
    self._V = self._V[np.argsort(angles)]

  def support(self, vector):
    return np.max(np.matmul(self.V, vector))
    # Optimization problem
    # result = linprog(-vector, A_ub = self.A, b_ub = self.b, bounds = (-np.inf, np.inf))
    # return -result.fun

  def translation(self, vector, substract=False):
    """Compute the polyhedron shifted by some vector"""
    if not isinstance(vector, (np.ndarray, list, tuple)):
      raise TypeError(f'translation() argument must be a numpy.ndarray (list), not {type(vector)}')
    # vector = np.array(vector, dtype=float)[:, np.newaxis]
    if vector.shape[0] != self.dim or vector.shape[1] != 1 or len(vector.shape) > 2:
      raise ValueError(f'The point must be a {self.dim}x1 vector')
    Ht = self.H
    Vt = self.V
    if substract:
      vector = -vector
    if self.V.shape[0] > 0:
      # V-rep: Just shift the vertices
      Vt = Vt + np.tile(vector.T, (Vt.shape[0],1))
    if self.H.shape[0] > 0:
      # H-rep: A*z <= b ==> A*y <= b + A*x
      bt = Ht[:,-1].reshape(Ht.shape[0], 1) + np.matmul(Ht[:,0:-1], vector)
      Ht = np.hstack((Ht[:,0:-1], bt))
    # print(Polyhedron(V=Vt, H=Ht))
    return Polyhedron(Vt) # Ht ??

  # @staticmethod
  # def fullSpace(dim):
  #   """Constructs the H-rep of R^n
  #   """
  #   # R^n is represented by 0'*x<=1
  #   P = Polyhedron(np.zeros(1,dim), 1)
  #   return P

  # def P_plus_p(P, point, subtract_p=False):
  #   # Polytope + point: The sum of a polytope in R^n and an n-vector -- P + p
  #   # If subtract_p == True, compute P - p instead. This implementation allows
  #   # writing Polytope(...) - (1.0, 0.5) or similar by overloading __sub__ with
  #   # this function.
  # p = np.array(np.squeeze(point), dtype=float)[:, np.newaxis]
  # if p.size != P.n or p.shape[1] != 1 or p.ndim != 2:  # ensure p is n x 1
  #   raise ValueError(f'The point must be a {P.n}x1 vector')
  # if subtract_p: p = -p # add -p if 'sub'
  # P_shifted = Polytope()
  # # V-rep: The sum is all vertices of P shifted by p.
  # # Not necessary to tile/repeat since p.T broadcasts, but could do something
  # # like np.tile(q.T, (P.nV, 1)) or np.repeat(p.T, P.nV, axis=0).
  # if P.in_V_rep:
  #   P_shifted.V = P.V + p.T
  #   # H-rep: A * x <= b shifted by p means that any point q = x + p in the shifted
  #   # polytope satisfies A * q <= b + A * p, since
  #   #   A * x <= b  ==>  A * (q - p) <= b  ==>  A * q <= b + A * p
  #   # That is, the shifted polytope has H-rep (A, b + A * p).
  # if P.in_H_rep:
  #   P_shifted._set_Ab(P.A, P.b + P.A @ p)  # TODO: redesign this
  # return P_shifted

# def linear_map(M, P):
#   # Compute the linear map M * P through the vertex representation of V. If P
#   # does not have a vertex representation, determine the vertex representation
#   # first. In this case, the H-representation of M * P is NOT determined before
#   # returning M * P.
#   n = M.shape[1]
#   if P.n != n:
#     raise ValueError('Dimension of M and P do not agree in linear map M * P')
#   # TODO: implement linear map in H-rep for invertible M
#   # TODO: M_P = Polytope(P.V @ M.T), if P.in_H_rep: M_P.determine_H_rep()?
#   return Polytope(P.V @ M.T)
