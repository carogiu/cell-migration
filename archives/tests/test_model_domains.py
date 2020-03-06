import unittest

import dolfin

from model.main import mesh_from_dim
from model.model_domains import BD_right, BD_left, BD_top_bottom, dom_and_bound


class TestModelDomains(unittest.TestCase):

    def test_class_BD(self):
        class_1 = BD_right(dim_x=2)
        self.assertIsInstance(class_1, BD_right)
        self.assertEqual(class_1.dim_x, 2)
        class_2 = BD_left(dim_x=2)
        self.assertIsInstance(class_2, BD_left)
        self.assertEqual(class_2.dim_x, 2)
        class_3 = BD_top_bottom(dim_y=2)
        self.assertIsInstance(class_3, BD_top_bottom)
        self.assertEqual(class_3.dim_y, 2)

    def test_class_BD_inside(self):
        class_2 = BD_right(dim_x=2)
        result_inside = class_2.inside(x=[1+1e-13], on_boundary=False)
        self.assertEqual(result_inside, False)
        result_inside = class_2.inside(x=[1+1e-13], on_boundary=True)
        self.assertEqual(result_inside, True)
        result_inside = class_2.inside(x=[1+1e-12], on_boundary=True)
        self.assertEqual(result_inside, True)
        result_inside = class_2.inside(x=[1+1e-11], on_boundary=True)
        self.assertEqual(result_inside, False)

    def test_mesh_definition(self):
        mesh = mesh_from_dim(nx=100, ny=100, dim_x=10, dim_y=10)
        self.assertIsInstance(mesh, dolfin.RectangleMesh)
        domain, boundaries = dom_and_bound(mesh, dim_x=10, dim_y=10)






if __name__ == '__main__':
    unittest.main()
