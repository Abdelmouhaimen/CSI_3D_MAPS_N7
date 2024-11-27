import unittest
import numpy as np
from decimate import Decimater
from obja import Face
from utilites import separate_upper_lower_parts

class TestDecimater(unittest.TestCase):
    def setUp(self):
        # Initialize a Decimater object and add simple vertices and faces
        self.decimater = Decimater()
        # Define simple vertices in a square (3D coordinates)
        self.decimater.vertices = [np.array([0.0, 0.0, 0.0]),
                                   np.array([1.0, 0.0, 0.0]),
                                   np.array([1.0, 1.0, 2.0]),
                                   np.array([0.0, 1.0, 0.0]),
                                   np.array([1.0, 2.0, 0.0]),
                                   np.array([2.0, 1.5, 0.0])]
        # Define faces for a square (2 triangles)
        self.decimater.faces = [Face(0, 1, 2), Face(0, 2, 3), Face(2,4,5), Face(2,1,5), Face(2,3,4)]
    
    def test_find_1ring_neighborhood(self):
        # Run function and check if neighbors are correct
        
        self.decimater.find_1ring_neighborhood()
        
        self.assertEqual(set(self.decimater.neighbor_faces[0]), {0, 1})
        self.assertEqual(set(self.decimater.neighbor_faces[2]), {0, 1,3,2,4})
        self.assertEqual(set(self.decimater.neighbor_vertices[0]), {1,2, 3})
        self.assertEqual(set(self.decimater.neighbor_vertices[2]), {0,1,3,4,5})

    def test_boundary_vertex(self) :
        self.decimater.find_1ring_neighborhood()
        r,_ = self.decimater.is_boundary_vertex(2)
        self.assertFalse(r)
        r,_ = self.decimater.is_boundary_vertex(1)
        self.assertTrue(r)
    
    def test_find_cyclique(self) :
        self.decimater.find_1ring_neighborhood()
        cycle = self.decimater.find_cyclique(2)
        self.assertEqual(cycle[0],[0,1,5,4,3,0])
        cycle = self.decimater.find_cyclique(1)
        
        self.assertEqual(cycle[0] , [0, 2, 5, 0])

     
    
    
    def test_compute_normal_vertex(self):
        # Run function and check if the normal is correctly calculated for a simple vertex
        self.decimater.find_1ring_neighborhood()
        normal = self.decimater.compute_normal_vertex(0)
        # Expected normal direction could be an arbitrary unit vector; verify direction
        np.testing.assert_almost_equal(np.linalg.norm(normal), 1.0)
    
    def test_calculate_curvature(self):
        # Run curvature calculation and check if result is sensible
        self.decimater.find_1ring_neighborhood()
        curvature = self.decimater.calculate_curvature(0)
        self.assertGreaterEqual(curvature, 0)
    
    def test_calculate_area(self):
        # Calculate area for a simple vertex
        self.decimater.find_1ring_neighborhood()
        area = self.decimater.calculate_area(0)
        self.assertGreater(area, 0)
        self.assertAlmostEqual(area , 2.23,delta= 0.1)
        
    
    
    def test_calculate_priority(self):
        # Run priority calculation and check if vertices are prioritized correctly
        self.decimater.find_1ring_neighborhood()
        self.decimater.calculate_priority()
        self.assertEqual(self.decimater.deleted_vertices, [5,3])
    """
    def test_remove_vertex_and_retriangulate(self):
        # Remove a vertex and verify retriangulation for continuity
        self.decimater.find_1ring_neighborhood()
        new_faces = self.decimater.remove_vertex_and_retriangulate(0)
        self.assertGreater(len(new_faces), 0)"""
    
    def test_separate_upper_lower_parts(self):
        self.decimater.find_1ring_neighborhood()
        cycle ,isboundary= self.decimater.find_cyclique(2)
        if not isboundary:
            cycle.pop(0)
        else :
            cycle.pop(-1)
        print(cycle)
        upper , lower = separate_upper_lower_parts(cycle,[5,3])
        self.assertEqual(upper,[5,4,3])
        self.assertEqual(lower, [3,0,1,5])

    def test_detect_tagged_feature_lines(self) :
        self.decimater.find_1ring_neighborhood()
        
        
        self.assertEqual(self.decimater.tagged_edges,{(2, 4), (1, 2), (2, 3), (0, 2), (2, 5)})





# Run the tests
if __name__ == '__main__':
    unittest.main()