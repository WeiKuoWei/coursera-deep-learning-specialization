import json
import re


def deploy_assignment(input, keep_unit_tests=True):
    ''' Creates the Assignment and the Solution notebooks from a development notebook.
    The function will create new files in the same folder where the input is located.
    Args:
        'input' (str): The path to the _dev.ipynb notebook.
        'keep_unit_tests' (bool): Keep unit tests in the assignments notebook?
    Examples:
        'deploy_assignment("C1_W1_Assignment_dev.ipynb", True)'
        # Produces: 'C1_W1_Assignment.ipynb'
                    'C1_W1_Assignment_Solution.ipynb'
    '''

    try:
        with open(input) as f:
            nb_solution = json.load(f)

        with open(input) as f:
            nb_assignment = json.load(f)

        # The special marks @ASSIGNMENT, @SOLUTION and @UNIT_TEST
        # must be at the first or second lines
        # For the solution notebook
        cells = nb_solution['cells']
        for cell in cells:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if source.find('# @ASSIGNMENT') >= 0:
                    cells.remove(cell)

        if (input.find('_Dev')):
            with open(input.replace('_Dev', '_Solution'), 'w') as file:
                file.write(json.dumps(nb_solution, indent=2))
        else:
            with open(input.replace('.ipynb', '_Solution.ipynb'), 'w') as file:
                file.write(json.dumps(nb_solution, indent=2))

        # For the assignment notebook
        cells2 = nb_assignment['cells']
        for cell in cells2:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if source.find('# @SOLUTION') >= 0:
                    cells2.remove(cell)
                else:
                    if not keep_unit_tests:
                        if source.find('# @UNIT_TEST') >= 0:
                            cells2.remove(cell)
        if (input.find('_Dev')):
            with open(input.replace('_Dev', '_'), 'w') as file:
                file.write(json.dumps(nb_assignment, indent=2))
        else:
            with open(input.replace('.ipynb', '_Assignment.ipynb'), 'w') as file:
                file.write(json.dumps(nb_assignment, indent=2))

    except ValueError:
        print("Error during deployment")
        print(ValueError)


