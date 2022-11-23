import re


def convert_string(inp_string):
    x = re.findall(r'(?<=[a-z])([A-Z])',inp_string)
    for c in x:
        idx = inp_string.index(c)
        inp_string = inp_string[:idx] + '_' + c.lower() + inp_string[idx+1:]
    return inp_string


def convert_py_line(py_line):
    if py_line.startswith('#') or py_line.startswith('class') or 'import' in py_line:
        return py_line
    return convert_string(py_line)


def convert_file(filename):
    with open(filename) as f:
        py_lines = f.readlines()

    converted_lines = [convert_py_line(pyl) for pyl in py_lines]

    with open(filename,'w') as f:
        for cpyl in converted_lines:
            f.write(cpyl)


convert_file('sem_type.py')
