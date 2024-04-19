# -*- coding: utf-8 -*-

"""
## Python (sub)module geometry
"""


import sys

from wiptools import get_workspace_dir

sys.path.insert(0, str(get_workspace_dir(__file__) / 'exponential_decay'))

from exponential_decay import generate_square_txy


if __name__ == '__main__':
    t,x,y = generate_square_txy(1,1,1)
    print('-*# finished #*-')

