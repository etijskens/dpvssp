# -*- coding: utf-8 -*-

"""
## Python (sub)module timing_tools
"""
import sys
from time import perf_counter_ns


class RuntimeTable:
    def __init__(self):
        columns = ['description', 'size', 's']
        self.widths = [0] * len(columns)
        self.data = []
        self.append(columns)

    @property
    def ncols(self):
        return len(self.data[0])

    @property
    def nrows(self):  # including column names
        return len(self.data)

    def format(self, c):
        sc = c if isinstance(c, str) else \
            str(c) if isinstance(c, (int, tuple)) else \
                f"{c:.3g}"
        return sc

    def append(self, row):
        """Append a row"""
        for ic, c in enumerate(row):
            self.widths[ic] = max(self.widths[ic], len(self.format(c)))

        self.data.append(row)

    def print(self):
        s = ''
        for i, row in enumerate(self.data):
            s += '\t '
            for c, w in zip(row[:-1], self.widths[:-1]):
                fc = self.format(c)
                s += f'{fc:{w}} | '
            fc = self.format(row[-1])
            s += f'{fc:{self.widths[-1]}}\n'
            if i == 0:
                s += '\t-'
                for w in self.widths[:-1]:
                    s += w * '-' + '-+-'
                s += self.widths[-1] * '-' + '-\n'
        print(s)

    def add_performance(self):
        self.data[0].append('eval/s')
        self.widths.append(6)
        for i in range(1, self.nrows):
            size = self.data[i][1]
            self.data[i].append(size / self.data[i][2])
            self.widths[-1] = max(self.widths[-1], len(self.format(self.data[i][-1])))

    def add_speedup(self):
        """Add columnn for speedup float32/float64"""
        self.data[0].append('SP/DP')
        self.widths.append(5)
        for ir in range(1, self.nrows, 2):
            self.data[ir].append('')
            speedup = self.data[ir + 1][3] / self.data[ir][3]
            self.data[ir + 1].append(speedup)
            self.widths[-1] = max(self.widths[-1], len(self.format(speedup)))


def time_fun(fun, runtime_table, description, size, repetitions=10, **kwargs):
    """Time the function `fun(**kwargs)` with `repetitions` repetitions and
    add the results to `runtime_table`."""

    rt = 1e24
    for r in range(repetitions):
        tic = perf_counter_ns()

        result = fun(**kwargs)

        toc = perf_counter_ns() - tic
        rt = min(rt, toc)
        print(f"{description} {size=}, {r}/{repetitions}: {toc=} min={rt}", file=sys.stderr)

    runtime_table.append([description, size, rt * 1e-9])  # ns -> s conversion


