from testing_utils import *
import unittest
import numpy as np

# Mainteiner: Andres Castillo
# Owner: DeepLearning.AI

class TestingUtilsTestCase(unittest.TestCase):

    def test_datatype_check(self):
        assert datatype_check("abc", "csf", "Error") == 1
        assert datatype_check([1, 2, 3], [4, 3, 5], "Error") == 1
        assert datatype_check([1, 2, 3], [4., 3., 5.], "Error", level=1) == 0
        assert datatype_check([], [1, 2, 3], "Error") == 1
        assert datatype_check([1, 2, 3], [], "Error") == 1
        assert datatype_check(np.array([1, 2, 3]), [], "Error", level=1) == 0
        assert datatype_check([1, 2, 3], [], "Error", level=1) == 0
        assert datatype_check(np.array([1, 2, 3]), [1, 2, 3], "Error") == 0
        assert datatype_check([1, 2, 3], np.array([1, 2, 3]), "Error") == 0

            
    def test_equation_output_check(self):
        assert equation_output_check("abc", "abc", "Error") == 1
        assert equation_output_check([1, 2, 3], [1, 2, 3], "Error") == 1
        assert equation_output_check([[1, 2, 3], [4, 5, 6], [9., 10., 11.]], np.array([[1., 2, 3.], [4, 5, 6], [9, 10, 11]]), "Error") == 1
        assert equation_output_check((1.0, 2.0, 3), [1, 2, 3], "Error") == 1
        assert equation_output_check([1, 2, 3], [], "Error") == 0
        assert equation_output_check([1, 2, 3], [2., 3., 4.], "Error") == 0
        assert equation_output_check(np.array([1, 2, 3]), [1, 2., 3], "Error") == 1
        assert equation_output_check(np.array([]), [1, 2, 3], "AssertionError 1") == 0
        assert equation_output_check([], [1, 2, 3], "AssertionError 1") == 0
        assert equation_output_check("abc", [1, 2, 3], "AssertionError 1") == 0
        assert equation_output_check([1, 2, 3], "abc",  "AssertionError 1") == 0
        assert equation_output_check("abc", "xyz", "AssertionError 2") == 0

    def test_shape_check(self):
        assert shape_check("abc", "csf", "Error") == 1 # It does not have sence, but I include it as reference
        assert shape_check("abc", "csfsss", "Error") == 1 # It does not have sence, but I include it as reference
        assert shape_check([1, 2, 3], [4, 3, 5], "Error") == 1
        assert shape_check(np.array([1, 2, 3]), np.array([4, 3, 5]), "Error") == 1
        assert shape_check([], [1, 2, 3], "Error") == 0
        assert shape_check([1, 2, 3], [], "Error") == 0
        assert shape_check([3, 6, 7], np.array([1, 2, 3]), "Error") == 1
        assert shape_check([[3, 2], [6, 0], [7, 7]], np.array([[3, 2], [6, 0], [7, 7]]), "Error") == 1
        assert shape_check(np.array([3, 6, 7]), [1, 2, 3], "Error") == 1
        assert shape_check(np.array([[3, 2], [6, 0], [7, 7]]), np.array([[3], [6], [7]]), "Error") == 0


if __name__ == '__main__':
    unittest.main()

