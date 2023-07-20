import os
import sys

module_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(module_dir, '../')
root_path = os.path.join(module_dir, '../../')

sys.path.append(lib_path)
sys.path.append(root_path)

