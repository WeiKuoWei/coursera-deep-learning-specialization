import json
import re
import sys

default_patterns = {
    'section': ['^## [0-9]+ - '], # '- [Part {section_number}: {section_title}](# {section_number})'
    'subsection': ['^### [0-9]+.[0-9]+ '], 
    'exercise': ['^### Exercise [0-9]+'], 
    'toc_title': ['^# Outline']
}

default_templates = {
    'section': '## {section_number} - {section_title}', 
    'subsection': '### {section_number}.{subsection_number} {subsection_title}', 
    'exercise': '### Exercise {exercise_number}', 
    'toc_title': '# Outline'
}

def add_toc(input, patterns=default_patterns, templates=default_templates):
    """ Function to include a table of content on a jupyter notebook based on its
        markdown. It receives a file path as input, and creates a new notebook
        file that includes the table of content in the same path.
        It performs some style fix during the process. For more information see
        the attached notebook example.

    Args:
        input ([strin]): The notebook file to be formated
    """
    try:
        with open(input) as f:
            notebook = json.load(f)

        cells = notebook['cells']

        head_cell = cells[0]

        variables = {'section_number': 0,
             'section_title': '',
             'subsection_number': 0,
             'subsection_title': '',
             'exercise_number': 0}

        toc = ['\n', templates['toc_title'] + '\n']

        for cell in cells[1: len(cells)]:
            if cell['cell_type'] == 'markdown':
                source_array = cell['source']
                row = 0
                skip = False
                for source in source_array:
                    print(f"{row} - {source}")
                    if not skip:
                        if re.search('^[#]+ Introduction', source, re.IGNORECASE):

                            html_tag = '<a name="0"></a>\n'
                            if row > 0 and source_array[row - 1].find('<a name=') >= 0:
                                source_array[row - 1] = html_tag
                            else:
                                source_array.insert(row, html_tag)
                                row += 1
                                skip = True

                            toc.append('- [Introduction](#0)\n')
                            source_array[row] = '# Introduction\n'
                            

                        if match_patterns(source, 'section', patterns):
                            variables['section_number'] += 1
                            variables['section_title'] = get_title(source, 'section', patterns)
                            html_tag = f'<a name="{variables["section_number"]}"></a>\n'

                            if row > 0 and source_array[row - 1].find('<a name=') >= 0:
                                source_array[row - 1] = html_tag
                            else:
                                source_array.insert(row, html_tag)
                                row += 1
                                skip = True

                            md_line = templates['section'].format(**variables)
                            toc.append(f'- [{md_line.replace("#", "")}](#{variables["section_number"]})\n')

                            # Fix source line if needed
                            source_array[row] = md_line + '\n'

                            variables['subsection_number'] = 0
                            #skip = True

                        if match_patterns(source, 'subsection', patterns):
                            variables['subsection_number'] += 1  # Increase section

                            label = f"{variables['section_number']}.{variables['subsection_number']}"
                            variables['subsection_title'] = get_title(source, 'subsection', patterns)
                            html_tag = f'<a name="{label}"></a>\n'
                            print(f"--- {row} - {source}")
                            print(f"{variables['section_number']}- {variables['subsection_number']}")
                            print(row)
                            if row > 0 and source_array[row - 1].find('<a name=') >= 0:
                                source_array[row - 1] = html_tag
                            else:
                                source_array.insert(row, html_tag)
                                row += 1
                                skip = True


                            md_line = templates['subsection'].format(**variables)
                            toc.append(f'  - [{md_line.replace("#", "")}](#{label})\n')
                            # Fix source line if needed
                            source_array[row] = f'{md_line}\n'

                        if match_patterns(source,'exercise', patterns):
                            variables['exercise_number'] += 1
                            label = "{:02d}".format(variables['exercise_number'])
                            html_tag = '<a name="ex' + label + '"></a>\n'

                            if row > 0 and source_array[row - 1].find('<a name=') >= 0:
                                source_array[row - 1] = html_tag
                            else:
                                source_array.insert(row, html_tag)
                                row += 1
                                skip = True

                            ident = '    '
                            if variables['subsection_number'] == 0:  # There is not subsection
                                ident = '  '

                            md_line = templates['exercise'].format(**variables)
                            toc.append(f'{ident}- [{md_line.replace("#", "")}](#ex{label})\n')

                            source_array[row] = f'{md_line}\n'
                            #skip = True
                    else:
                        skip = False
                        row -= 1
                    row += 1
            # Clear outputs and reset the execution count
            if cell['cell_type'] == 'code':
                cell['execution_count'] = 0
                cell['outputs'] = []

        # Remove the old toc
        toc_line = 0
        for line in head_cell['source']:
            if re.search('^' + patterns['toc_title'][0], line, re.IGNORECASE):
                head_cell['source'] = head_cell['source'][0: toc_line]
                break
            toc_line += 1

        head_cell['source'] = head_cell['source'] + toc

        with open(input.replace('.ipynb', '_toc.ipynb'), 'w') as file:
            file.write(json.dumps(notebook, indent=2))

    except ValueError:
        print("Error during deployment")
        print(ValueError)


def match_patterns(line, name, patterns):
    for pattern in patterns[name]:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def get_title(line, name, patterns):
    for pattern in patterns[name]:
        title_search = re.search(pattern, line, re.IGNORECASE)
        if title_search:
            return re.sub(pattern, '', line).replace('\n', '')
    return ''

if __name__ == "__main__":
    if sys.argv[1] == 'help':
        print(f'Usage: \n dlai_tools.add_toc input_file.ipynb [patterns.json templates.json]')
    else:
        patterns = default_patterns
        templates = default_templates
        if len(sys.argv) > 3 and '.json' in sys.argv[2]:
            with open(sys.argv[2]) as f:
                patterns = json.load(f)
            with open(sys.argv[3]) as f:
                templates = json.load(f)

        add_toc(sys.argv[1], patterns, templates)