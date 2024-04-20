# -*- coding: utf-8 -*-

"""
## Python script for timing the heat equation
"""

import sys
from wiptools import get_workspace_dir
sys.path.insert(0, str(get_workspace_dir(__file__) / 'dpvssp'))

from dpvssp.timing_tools import RuntimeTable, time_fun
from dpvssp.heat.geometry import generate_square
from dpvssp.heat import compute_Txy
from numpy import float32, float64


if __name__ == '__main__':
    runtime_table = RuntimeTable()
    experiment = 0

    for n in (4, 16, 64, 256):
        for blocked in (True, False):
            for float_type in (float64, float32):
                t,x,y = generate_square(n, experiment, float_type)
                description = f'{float_type.__name__}_{2*n+1}x{2*n+1}'
                if blocked:
                    description += '_blocked'

                time_fun( compute_Txy, runtime_table, description, len(x), repetitions=1
                        , t=t, x=x, y=y, T0=300, experiment=0, blocked=blocked
                        )

    runtime_table.add_performance()
    runtime_table.add_speedup()
    runtime_table.print()
    print('-*# finished #*-')
