import numpy as np
from antope import Polyhedron
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

polyhedron_square = Polyhedron(np.vstack((np.eye(2), -np.eye(2))), np.ones((4,1)))
polyhedron_square.plot(ax)

plt.autoscale()
plt.show()
