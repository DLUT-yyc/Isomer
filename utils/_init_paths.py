import os
import sys

tools_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(tools_dir, '..')
models_path = os.path.join(tools_dir, '../utils')

sys.path.append(tools_dir)
sys.path.append(base_path)
sys.path.append(models_path)

