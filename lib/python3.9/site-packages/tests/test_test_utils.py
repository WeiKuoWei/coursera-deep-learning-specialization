from dlai_tools.test_utils import *
import numpy as np


assert datatype_check("abc", "csf", "Error") == 1
assert datatype_check([1, 2, 3], [4, 3, 5], "Error") == 1
assert equation_output_check("abc", "abc", "Error") == 1
assert equation_output_check(np.array([1, 2, 3]), [1, 2, 3], "Error") == 1

print("All test passed")
#try:
#    datatype_check("abc", 1, "Error")
#except: 

#datatype_check([1, 2, 3], [4, 3, 5], "Error")
#equation_output_check("abc", "abc", "Error")
#equation_output_check(np.array([1, 2, 3]), [1, 2, 3], "Error")