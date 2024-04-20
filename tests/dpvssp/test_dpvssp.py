# -*- coding: utf-8 -*-

"""Tests for dpvssp package."""

import sys
from wiptools import get_workspace_dir
sys.path.insert(0, str(get_workspace_dir(__file__) / 'dpvssp'))

import dpvssp
from dpvssp.timing_tools import RuntimeTable

import numpy as np
from time import perf_counter_ns
from tqdm import tqdm

def test_generate_random_array():

    shape = 10

    result = dpvssp.generate_random_array(shape)
    assert result.dtype is np.dtype('float64')
    assert np.all(0. <= result) and np.all(result < 1.0)

    result = dpvssp.generate_random_array(shape, dtype=np.float32)
    assert result.dtype is np.dtype('float32')
    assert np.all(0. <= result) and np.all(result < 1.0)

    shape = (4,4)

    result = dpvssp.generate_random_array(shape)
    assert result.dtype is np.dtype('float64')
    assert np.all(0. <= result) and np.all(result < 1.0)

    result = dpvssp.generate_random_array(shape, dtype=np.float32)
    assert result.dtype is np.dtype('float32')
    assert np.all(0. <= result) and np.all(result < 1.0)


def test_sum_nxn():
    repetitions = 20

    n = 10
    for i in range(6):
        shape = (n,n)
        for dtype in (np.float64, np.float32):
            runtime = 1e12 
            for r in range(repetitions):
                # a = np.zeros(shape, dtype=dtype)
                a = dpvssp.generate_random_array(shape, dtype=dtype)

                tic = perf_counter_ns()
                sum = np.sum(a)
                toc = perf_counter_ns()

                rt = toc - tic
                runtime = min(runtime, rt)
            print(f"dtype={dtype.__name__} {n=} nxn={n}x{n}={n*n} {runtime=}ns")
        n *= 10

    
def test_sum_n():
    repetitions = 20
    for n in range(3,10):
        shape = (int(10**n),)
        time_fun(np.sum, repetitions=repetitions, shape=shape, verbose=True)


def test_time():
    repetitions = 50
    shapes = [(int(10**n),) for n in range(3,9)]
    funs = [np.add, np.multiply, np.divide]
    for fun in funs:
        print(f"\n{fun.__name__}(a,b)")
        runtime_table = time_fun(fun, repetitions=repetitions, shapes=shapes, n_arrays=2)
        runtime_table.print()

    funs = [np.exp, np.sqrt]
    for fun in funs:
        print(f"\n{fun.__name__}(a)")
        runtime_table = time_fun(fun, repetitions=repetitions, shapes=shapes, n_arrays=1)
        runtime_table.print()


def time_fun(fun, repetitions, shapes, n_arrays=1):
    """Time the function `fun` """

    runtimes = RuntimeTable()

    if not isinstance(shapes, list):
        shapes = [shapes]
    
    for shape in shapes:
        arrays = []
        for ia in range(n_arrays):
            arrays.append(dpvssp.generate_random_array(shape, dtype=np.float64))
        
        for dtype in (np.float64, np.float32):
            if dtype is np.float32:
                for ia,a in enumerate(arrays):
                    arrays[ia] = a.astype(np.float32)
            rt = 1e12 
            for r in tqdm(range(repetitions)):
                tic = perf_counter_ns()
                
                result = fun(*arrays)
                
                rt_r = perf_counter_ns() - tic
                rt = min(rt, rt_r)

            runtimes.append([dtype.__name__, shape, rt * 1e-9]) # ns -> s conversion
         
    runtimes.add_performance()
    runtimes.add_speedup()

    return runtimes
    

class RuntimeTable: 
    def __init__(self):
        columns = ['dtype','size', 's']
        self.widths = [0]*len(columns)
        self.data = []
        self.append(columns)
        
    @property
    def ncols(self):
        return len(self.data[0])

    @property
    def nrows(self): # including column names
        return len(self.data)
    
    def format(self, c):
        sc = c      if isinstance(c, str)          else \
             str(c) if isinstance(c, (int, tuple)) else \
             f"{c:.3g}"
        return sc

    def append(self, row):
        """Append a row"""
        for ic,c in enumerate(row):
            self.widths[ic] = max(self.widths[ic], len(self.format(c)))

        self.data.append(row)

    def print(self):
        s = ''
        for i,row in enumerate(self.data):
            s += '\t '
            for c,w in zip(row[:-1],self.widths[:-1]):
                fc = self.format(c)
                s += f'{fc:{w}} | '
            fc = self.format(row[-1])
            s += f'{fc:{self.widths[-1]}}\n'
            if i == 0:
                s +='\t-'
                for w in self.widths[:-1]:
                    s += w*'-' + '-+-'
                s += self.widths[-1]*'-' + '-\n'
        print(s)

    def add_performance(self):
        self.data[0].append('eval/s')
        self.widths.append(6)
        for i in range(1,self.nrows):
            shape = self.data[i][1]
            size = shape[0]
            for n in shape[1:]:
                size *= n
            self.data[i].append(size / self.data[i][2])
            self.widths[-1] = max(self.widths[-1], len(self.format(self.data[i][-1])))
    
    def add_speedup(self):
        """Add columnn for speedup float32/float64"""
        self.data[0].append('SP/DP')
        self.widths.append(5)
        for ir in range(1,self.nrows,2):
            self.data[ir]  .append('')
            speedup = self.data[ir+1][3] / self.data[ir][3]
            self.data[ir+1].append( speedup )
            self.widths[-1] = max(self.widths[-1], len(self.format(speedup)))


def test_RuntimeTable():
    tbl = RuntimeTable()
    tbl.append([np.float64.__name__, 10, float(100)])
    tbl.print()
    tbl.add_performance()
    tbl.print()


def heat_equation(a, t1, c, T, w2=0.111, Ethpi32rho=0.111, md2=0.111):
    """
    Args:
        a : array
        t1: array
        c : array
        T : array
        w2: scalar
        Ethpi32rho: scalar
        md2: scalar
    """
    # simple numpy array operations (involving temporary arrays.)
    a4dt   = (4*a)*t1
    a4dtw2 = a4dt + w2

    Trise = ( (Ethpi32rho / c)
            / (np.sqrt(a4dt) * a4dtw2)
            ) * np.exp( md2 / a4dtw2)
    T += Trise


def test_time_heat_equation():
    repetitions = 50
    shapes = [(int(10**n),) for n in range(4,6)]
    funs = [heat_equation]
    for fun in funs:
        print(f"\n\t{fun.__name__}(a,t1,b,T) {repetitions=}")
        runtime_table = time_fun(fun, repetitions=repetitions, shapes=shapes, n_arrays=4)
        runtime_table.print()



# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_time_heat_equation

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print('-*# finished #*-')

# eof