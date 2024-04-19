# -*- coding: utf-8 -*-
"""
## Python (sub)module process

read process parameters from file.
"""

import sys

from wiptools import get_workspace_dir

sys.path.insert(0, str(get_workspace_dir(__file__) / 'exponential_decay'))

from exponential_decay import get_fluence, get_omega0, get_Eth_omega0, \
                              get_laser_frequency, get_mark_speed


if __name__ == '__main__':
    Eth, omega0 = get_Eth_omega0(1)
    print('-*# finished #*-')