# -*- coding: utf-8 -*-

"""
## Python (sub)module geometry
"""


import sys

import numpy as np
from wiptools import get_workspace_dir

sys.path.insert(0, str(get_workspace_dir(__file__) / 'exponential_decay'))

from exponential_decay import generate_square_txy as _generate_square_txy
import dpvssp.heat.process as process

def generate_square(n, experiment, float_type=np.float64):
    """Generate (2n+1)x(2n+1) square of pulses"""

    deltat = 1./process.get_laser_frequency(experiment)
    deltax = process.get_mark_speed(experiment)*deltat

    t,x,y = _generate_square_txy(n,deltat,deltax)

    if float_type is np.float64:
        pass
    elif float_type is np.float32:
        t = t.astype(float_type)
        x = x.astype(float_type)
        y = y.astype(float_type)
    else:
        raise NotImplementedError(f"Unsupported float type {float_type}")

    return t,x,y


if __name__ == '__main__':
    t,x,y = generate_square(1,0)
    print('-*# finished #*-')

