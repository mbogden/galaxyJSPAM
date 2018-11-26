import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

__all__ = ['Structure',
           ]

from structure import Structure
from get_target_data import get_target_data
