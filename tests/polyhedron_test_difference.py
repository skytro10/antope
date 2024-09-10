import numpy as np
from antope import Polyhedron

polyhedron_a = Polyhedron(np.vstack((np.eye(2), -np.eye(2))), np.ones((4,1)))
polyhedron_b = Polyhedron(np.vstack((np.eye(2), -np.eye(2))), 0.5 * np.ones((4,1)))
polyhedron_d = polyhedron_b - polyhedron_a

polyhedron_d.V
print(polyhedron_d.V)
