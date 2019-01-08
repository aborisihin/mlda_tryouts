""" Settings reader script. Used for reading JSON values for makefile.
Using: python settings_reader.py <settings path> <path in json tree with '/' separator>
"""

import sys
import json

if __name__ == '__main__':
    param_value = ''
    if len(sys.argv) >= 3:
        with open(sys.argv[1], 'r') as param_file:
            params = json.load(param_file)
            try:
                # разбор пути в дереве
                tree_path = str(sys.argv[2]).split('/')
                param_value = params
                for node in tree_path:
                    param_value = param_value.get(node)
            except KeyError:
                param_value = ''
    sys.stdout.write('{}\n'.format(param_value))
