from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Conv1D

import numpy as np
from termcolor import colored
from copy import deepcopy

# Compare the two inputs
def comparator(learner, instructor):
    layer = 0
    if len(learner) != len(instructor):
        raise AssertionError(f"The number of layers in the model is incorrect. Expected: {len(instructor)} Found: {len(learner)}") 
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  f"at layer: {layer}",
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
        layer += 1
    print(colored("All tests passed!", "green"))
    return True

# extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
        if (type(layer) == Conv1D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.strides)
            descriptors.append(layer.kernel_size)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
           
        if (type(layer) == Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
            
        if (type(layer) == MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
            
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
            
        if (type(layer) == ZeroPadding2D):
            descriptors.append(layer.padding)
            
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__)
            
        if (type(layer) == LSTM):
            descriptors.append(layer.input_shape)
            descriptors.append(layer.activation.__name__)
            
        if (type(layer) == RepeatVector):
            descriptors.append(layer.n)
            
        if (type(layer) == TimeDistributed):
            descriptors.append(layer.layer.activation.__name__)  
            
        if (type(layer) == GRU):
            descriptors.append(layer.return_sequences)    
            
        result.append(descriptors)
    return result


def datatype_check(expected_output, target_output, error, level=0):
    success = 0
    if (level == 0):
        try:
            assert isinstance(target_output, type(expected_output))
            return 1
        except:
            return 0
    else:
        if isinstance(expected_output, tuple) or isinstance(expected_output, list) \
                or isinstance(expected_output, np.ndarray) or isinstance(expected_output, dict):
            if isinstance(expected_output, dict):
                range_values = expected_output.keys()
            else:
                range_values = range(len(expected_output))
            if len(expected_output) != len(target_output) or not isinstance(target_output, type(expected_output)):
                return 0
            for i in range_values:
                try:
                    success += datatype_check(expected_output[i],
                                            target_output[i], error, level - 1)
                except:
                    print("Error: {} in variable {}, expected type: {}  but expected type {}".format(error,
                                                                                                    i,
                                                                                                    type(
                                                                                                        target_output[i]),
                                                                                                    type(expected_output[i]
                                                                                                        )))
            if success == len(expected_output):
                return 1
            else:
                return 0

        else:
            try:
                assert isinstance(target_output, type(expected_output))
                return 1
            except:
                return 0


def equation_output_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, tuple) or isinstance(expected_output, list) or isinstance(expected_output, dict):
        if isinstance(expected_output, dict):
            range_values = expected_output.keys()
        else:
            range_values = range(len(expected_output))

        if len(expected_output) != len(target_output):
                return 0

        for i in range_values:
            try:
                success += equation_output_check(expected_output[i],
                                                 target_output[i], error)
            except:
                print("Error: {} for variable in position {}.".format(error, i))
        if success == len(expected_output):
            return 1
        else:
            return 0

    else:
        try:
            if hasattr(expected_output, 'shape'):
                np.testing.assert_array_almost_equal(
                    target_output, expected_output)
            else:
                assert target_output == expected_output
        except:
            return 0
        return 1


def shape_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, tuple) or isinstance(expected_output, list) or \
            isinstance(expected_output, dict) or isinstance(expected_output, np.ndarray):
        if isinstance(expected_output, dict):
            range_values = expected_output.keys()
        else:
            range_values = range(len(expected_output))

        if len(expected_output) != len(target_output):
                return 0
        for i in range_values:
            try:
                success += shape_check(expected_output[i],
                                       target_output[i], error)
            except:
                print("Error: {} for variable {}.".format(error, i))
        if success == len(expected_output):
            return 1
        else:
            return 0

    else:
        return 1


def single_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            if test_case['name'] == "datatype_check":
                assert isinstance(target(*test_case['input']),
                                  type(test_case["expected"]))
                success += 1
            if test_case['name'] == "equation_output_check":
                assert np.allclose(test_case["expected"],
                                   target(*test_case['input']))
                success += 1
            if test_case['name'] == "shape_check":
                assert test_case['expected'].shape == target(
                    *test_case['input']).shape
                success += 1
        except:
            print("Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))


def multiple_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            test_input = deepcopy(test_case['input'])
            target_answer = target(*test_input)
        except:
            print('\33[30m', "Error, interpreter failed when running test case with these inputs: " + 
                  str(test_input))
            raise AssertionError("Unable to successfully run test case.".format(target.__name__))

        try:
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'],
                                      target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(
                    test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'],
                                   target_answer, test_case['error'])
        except:
            print('\33[30m', "Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))
