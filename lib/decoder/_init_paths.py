import os
import sys

decoder_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(decoder_dir, '../')
root_path = os.path.join(decoder_dir, '../../')

sys.path.append(lib_path)
sys.path.append(root_path)

