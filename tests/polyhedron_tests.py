import unittest
import numpy as np
from antope import Polyhedron


class PolyhedronTestCase(unittest.TestCase):

    def test_instantiation(self):
        """
        Test different possible instantiations of a Polyhedron object.
        """
        polyhedron_01 = Polyhedron(1)
        self.assertEqual(polyhedron_01.dim, 1)
        self.assertTrue(polyhedron_01.has_vrep)
        self.assertFalse(polyhedron_01.has_hrep)
        self.assertFalse(polyhedron_01.is_empty)
        self.assertTrue(polyhedron_01)
        
        polyhedron_02 = Polyhedron(H = np.array([[1, 1]]))
        self.assertEqual(polyhedron_02.dim, 1)
        self.assertTrue(polyhedron_02.has_hrep)
        self.assertFalse(polyhedron_02.has_vrep)
        self.assertFalse(polyhedron_02.is_empty)
        self.assertTrue(polyhedron_02)
        
        polyhedron_03 = Polyhedron()
        self.assertEqual(polyhedron_03.dim, 0)
        self.assertFalse(polyhedron_03.has_hrep)
        self.assertFalse(polyhedron_03.has_vrep)
        self.assertTrue(polyhedron_03.is_empty)
        self.assertFalse(polyhedron_03)


if __name__ == '__main__':
    unittest.main()
