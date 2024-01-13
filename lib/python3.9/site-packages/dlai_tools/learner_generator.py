import json
import re
import sys


defaul_tags =  {
    'tagStartCode': "START CODE HERE",
    'tagEndCode': "END CODE HERE",
	'tagStartOmitBlock': "START OMIT BLOCK",
	'tagEndOmitBlock': "END OMIT BLOCK",
	'tagOmit': "@OMIT",
	'tagKeep': "@KEEP",
	'tagReplaceEquals': "@REPLACE EQUALS",
	'tagReplace': "@REPLACE",
}

def learner_generator(input, tags = defaul_tags, keep_unit_tests=True):
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
            nb_assignment = json.load(f)

        # For the assignment notebook
        cells2 = nb_assignment['cells']
        for cell in cells2:
            code_block = False
            omit_block = False
            if cell['cell_type'] == 'code':
                source = cell['source']
                line_number = -1
                for ln in source:
                    line_number += 1
                    if tags['tagStartCode'] in ln:
                        code_block = True
                        next

                    if tags['tagEndCode'] in ln:
                        code_block = False
                        next

                    if tags['tagStartOmitBlock'] in ln:
                        omit_block = True
                        next

                    if tags['tagEndOmitBlock'] in ln:
                        omit_blocklock = False
                        next

                    if omit_block or tags['tagOmit'] in ln:
                        next
                    
                    if code_block and not omit_block:
                        # Example: 'k = b[10] # @KEEP'   => 'k = b[10]'
                        if tags['tagKeep'] in ln:
                            source[line_number] = re.split("#[\ ]*"+tags['tagKeep'], ln)[0]
                            continue
                        # Example: 'k = b[10] # @REPLACE EQUALS b[None]'   => 'k = b[None]'
                        if tags['tagReplaceEquals'] in ln:
                            splitted = re.split("#[\ ]*" + tags['tagReplaceEquals'] + '[\ ]*', ln)
                            new_val = splitted[1]
                            splitted_eq = splitted[0].split("=")
                            source[line_number] = splitted_eq[0] + "= " + new_val
                            continue
                        # Example: 'k = b[10] # @REPLACE x = b[None]'   => 'x = b[None]'
                        if tags['tagReplace'] in ln:
                                ind = get_indentation(ln)
                                splitted = re.split("#[\ ]*" + tags['tagReplace'] + '[\ ]*', ln)
                                source[line_number] =  ind + splitted[1]
                                continue
                        # Keep lines starting with for, while, if
                        if 'for ' in ln or 'while ' in ln or 'if ' in ln:
                            continue 
                        # Example: 'k = b[10] # Comment'   => 'k = None'
                        if "=" in ln:
                                splitted = re.split("=", ln)
                                source[line_number] = splitted[0] + "= None\n"
                                continue
        print('dsdsdd')
        if '_Dev'in input:
            with open(input.replace('_Dev', ''), 'w') as file:
                file.write(json.dumps(nb_assignment, indent=2))
        else:
            with open(input.replace('.ipynb', '_Assignment.ipynb'), 'w') as file:
                file.write(json.dumps(nb_assignment, indent=2))

    except:
        print("Error during deployment")
        #print(ValueError)

def get_indentation(ln):
  try:
    return re.search('^[\ ]+', ln).group(0)
  except:
    return ''

if __name__ == "__main__":
    if sys.argv[1] == 'help':
        print(f'Usage: \n dlai_tools.learner_generator input_file.ipynb [tags.json]')
    else:
        tags = defaul_tags
        if len(sys.argv) > 2 and '.json' in sys.argv[2]:
            with open(sys.argv[2]) as f:
                tags = json.load(f)
        learner_generator(sys.argv[1], tags)