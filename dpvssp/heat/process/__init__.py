# -*- coding: utf-8 -*-

"""
## Python (sub)module process

read process parameters from file.
"""

from pathlib import Path
import sys

p = Path(__file__).parent.parent.parent.parent.parent / 'exponential_decay'
print(p.absolute())
sys.path.insert(0, str(p))

from exponential_decay import get_fluence, get_omega0, get_Eth_omega0, \
                              get_laser_frequency, get_mark_speed


if __name__ == '__main__':
    get_Eth_omega0(1)
    print('-*# finished #*-')