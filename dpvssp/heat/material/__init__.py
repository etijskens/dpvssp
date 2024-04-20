# -*- coding: utf-8 -*-

"""
## Python (sub)module material

read process parameters from file.
"""

import sys
from wiptools import get_workspace_dir

sys.path.insert(0, str(get_workspace_dir(__file__) / 'exponential_decay'))

from exponential_decay import rho, get_material_polynomials, get_material_slopes


if __name__ == '__main__':
    a,c = get_material_polynomials()
    print('-*# finished #*-')
