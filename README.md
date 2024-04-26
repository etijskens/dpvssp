# Python package dpvssp - experimenting with SP vs DP numpy arrays

The expectation that single precision arithmetic is in general twice as fast as double precision stems from the fact that

* single precision data occupies half the amount of bytes in mainn memory, cache and registers,
* vector registers therefore containt twice as much data items,
* the bandwith in terms of data items per second is twice as big.

Yet, in a previous project (https://github.com/etijskens/exponential_decay) we struggled with achieving better performance in single precision (SP) than in double precision (DP). The setting was a Python/ program evaluating a function over a large number of points, stored as numpy arrays and using the numba.vectorize decorator.

GJB reported in that respect:

    Er zijn inderdaad situaties waar de rekentijd niet echt wezenlijk verschilt.  Als ik en (sic) np.sum doe op een np.float32 en een np.float64 matrix krijg ik 20 %, geen factor 2.  (Mijn machine, skylake CPU.)

This project will carry out experiments to quantify these expectations.

Here are some results for some numpy array operators that were of interest in the previous project, addition, multiplication, division, exponentiation and square root. All operators show a performance ratio `SP/DP` of about 2, for large enough array sizes, except exponentiation, which has a performance ratio of 1. In addition it is 6 times slower than the square root operator.

    [Results on Apple M3 Pro]

    numpy.add(a,b)
     dtype   | size         | s        | eval/s   | SP/DP
    ---------+--------------+----------+----------+-------
     float64 | (1000,)      | 4.16e-07 | 2.4e+09  |     
     float32 | (1000,)      | 3.75e-07 | 2.67e+09 | 1.11
     float64 | (10000,)     | 2.67e-06 | 3.75e+09 |     
     float32 | (10000,)     | 9.58e-07 | 1.04e+10 | 2.78
     float64 | (100000,)    | 2.07e-05 | 4.83e+09 |     
     float32 | (100000,)    | 1.05e-05 | 9.49e+09 | 1.96
     float64 | (1000000,)   | 0.00049  | 2.04e+09 |     
     float32 | (1000000,)   | 0.000199 | 5.03e+09 | 2.47
     float64 | (10000000,)  | 0.00563  | 1.78e+09 |     
     float32 | (10000000,)  | 0.00269  | 3.72e+09 | 2.1 
     float64 | (100000000,) | 0.0732   | 1.37e+09 |     
     float32 | (100000000,) | 0.0317   | 3.16e+09 | 2.31
    
    numpy.multiply(a,b)
     dtype   | size         | s        | eval/s   | SP/DP
    ---------+--------------+----------+----------+-------
     float64 | (1000,)      | 4.58e-07 | 2.18e+09 |     
     float32 | (1000,)      | 4.17e-07 | 2.4e+09  | 1.1 
     float64 | (10000,)     | 2.38e-06 | 4.21e+09 |     
     float32 | (10000,)     | 9.16e-07 | 1.09e+10 | 2.59
     float64 | (100000,)    | 2.08e-05 | 4.82e+09 |     
     float32 | (100000,)    | 1.05e-05 | 9.49e+09 | 1.97
     float64 | (1000000,)   | 0.000487 | 2.05e+09 |     
     float32 | (1000000,)   | 0.000199 | 5.03e+09 | 2.45
     float64 | (10000000,)  | 0.00557  | 1.8e+09  |     
     float32 | (10000000,)  | 0.00301  | 3.32e+09 | 1.85
     float64 | (100000000,) | 0.0724   | 1.38e+09 |     
     float32 | (100000000,) | 0.0316   | 3.17e+09 | 2.29
     
    numpy.divide(a,b)
     dtype   | size         | s        | eval/s   | SP/DP
    ---------+--------------+----------+----------+-------
     float64 | (1000,)      | 5.41e-07 | 1.85e+09 |     
     float32 | (1000,)      | 4.58e-07 | 2.18e+09 | 1.18
     float64 | (10000,)     | 2.29e-06 | 4.36e+09 |     
     float32 | (10000,)     | 9.58e-07 | 1.04e+10 | 2.39
     float64 | (100000,)    | 2.03e-05 | 4.93e+09 |     
     float32 | (100000,)    | 1.03e-05 | 9.76e+09 | 1.98
     float64 | (1000000,)   | 0.000589 | 1.7e+09  |     
     float32 | (1000000,)   | 0.00029  | 3.45e+09 | 2.03
     float64 | (10000000,)  | 0.00566  | 1.77e+09 |     
     float32 | (10000000,)  | 0.00299  | 3.34e+09 | 1.89
     float64 | (100000000,) | 0.0727   | 1.38e+09 |     
     float32 | (100000000,) | 0.0318   | 3.14e+09 | 2.29
     
    numpy.exp(a)
     dtype   | size         | s        | eval/s   | SP/DP
    ---------+--------------+----------+----------+-------
     float64 | (1000,)      | 1.83e-06 | 5.45e+08 |     
     float32 | (1000,)      | 1.83e-06 | 5.46e+08 | 1   
     float64 | (10000,)     | 1.63e-05 | 6.12e+08 |     
     float32 | (10000,)     | 1.63e-05 | 6.14e+08 | 1   
     float64 | (100000,)    | 0.000161 | 6.23e+08 |     
     float32 | (100000,)    | 0.000161 | 6.21e+08 | 0.998
     float64 | (1000000,)   | 0.00183  | 5.47e+08 |     
     float32 | (1000000,)   | 0.0017   | 5.88e+08 | 1.08
     float64 | (10000000,)  | 0.019    | 5.28e+08 |     
     float32 | (10000000,)  | 0.0178   | 5.62e+08 | 1.07
     float64 | (100000000,) | 0.212    | 4.71e+08 |     
     float32 | (100000000,) | 0.183    | 5.46e+08 | 1.16
     
    numpy.sqrt(a)
     dtype   | size         | s        | eval/s   | SP/DP
    ---------+--------------+----------+----------+-------
     float64 | (1000,)      | 5.83e-07 | 1.72e+09 |     
     float32 | (1000,)      | 4.16e-07 | 2.4e+09  | 1.4 
     float64 | (10000,)     | 2.71e-06 | 3.69e+09 |     
     float32 | (10000,)     | 1.5e-06  | 6.67e+09 | 1.81
     float64 | (100000,)    | 2.5e-05  | 4e+09    |     
     float32 | (100000,)    | 1.29e-05 | 7.74e+09 | 1.94
     float64 | (1000000,)   | 0.00049  | 2.04e+09 |     
     float32 | (1000000,)   | 0.000212 | 4.72e+09 | 2.31
     float64 | (10000000,)  | 0.00533  | 1.87e+09 |     
     float32 | (10000000,)  | 0.00276  | 3.62e+09 | 1.93
     float64 | (100000000,) | 0.0724   | 1.38e+09 |     
     float32 | (100000000,) | 0.0308   | 3.24e+09 | 2.35

See [this stackoverflow issue](https://stackoverflow.com/questions/77028828/why-are-numpy-operations-with-float32-significantly-faster-than-float64), as to why numpy exponentation is not faster in single precision.

On Vaughan compute nodes the results are similar with the notable exception that now SP exponentiation is around 5 times faster than DP exponentiation. This suggests that SP exponentiation is vectorised and DP exponentiation not. 

However, it does not explain the observation in the [exponential_decay project](https://github.com/etijskens/exponential_decay) that replacing DP with SP arrays did not improve the performance.

    [results on Vaughan (compute node)]

    add(a,b)
	 dtype   | size         | s        | eval/s   | SP/DP
	---------+--------------+----------+----------+-------
	 float64 | (1000,)      | 1.71e-06 | 5.85e+08 |      
	 float32 | (1000,)      | 1.47e-06 | 6.8e+08  | 1.16 
	 float64 | (10000,)     | 4.76e-06 | 2.1e+09  |      
	 float32 | (10000,)     | 3.13e-06 | 3.2e+09  | 1.52 
	 float64 | (100000,)    | 4.4e-05  | 2.27e+09 |      
	 float32 | (100000,)    | 2.21e-05 | 4.52e+09 | 1.99 
	 float64 | (1000000,)   | 0.00113  | 8.81e+08 |      
	 float32 | (1000000,)   | 0.000252 | 3.97e+09 | 4.51 
	 float64 | (10000000,)  | 0.0149   | 6.7e+08  |      
	 float32 | (10000000,)  | 0.00785  | 1.27e+09 | 1.9  
	 float64 | (100000000,) | 0.143    | 7.01e+08 |      
	 float32 | (100000000,) | 0.0717   | 1.39e+09 | 1.99 


    multiply(a,b)
	 dtype   | size         | s        | eval/s   | SP/DP
	---------+--------------+----------+----------+-------
	 float64 | (1000,)      | 1.71e-06 | 5.85e+08 |      
	 float32 | (1000,)      | 1.45e-06 | 6.9e+08  | 1.18 
	 float64 | (10000,)     | 4.77e-06 | 2.1e+09  |      
	 float32 | (10000,)     | 3.06e-06 | 3.27e+09 | 1.56 
	 float64 | (100000,)    | 4.38e-05 | 2.29e+09 |      
	 float32 | (100000,)    | 2.2e-05  | 4.54e+09 | 1.99 
	 float64 | (1000000,)   | 0.00105  | 9.53e+08 |      
	 float32 | (1000000,)   | 0.000249 | 4.02e+09 | 4.21 
	 float64 | (10000000,)  | 0.015    | 6.67e+08 |      
	 float32 | (10000000,)  | 0.00788  | 1.27e+09 | 1.9  
	 float64 | (100000000,) | 0.143    | 6.98e+08 |      
	 float32 | (100000000,) | 0.0719   | 1.39e+09 | 1.99 


    divide(a,b)
	 dtype   | size         | s        | eval/s   | SP/DP
	---------+--------------+----------+----------+-------
	 float64 | (1000,)      | 1.92e-06 | 5.21e+08 |      
	 float32 | (1000,)      | 1.5e-06  | 6.67e+08 | 1.28 
	 float64 | (10000,)     | 6.68e-06 | 1.5e+09  |      
	 float32 | (10000,)     | 3.25e-06 | 3.08e+09 | 2.06 
	 float64 | (100000,)    | 5.56e-05 | 1.8e+09  |      
	 float32 | (100000,)    | 2.21e-05 | 4.52e+09 | 2.51 
	 float64 | (1000000,)   | 0.00106  | 9.44e+08 |      
	 float32 | (1000000,)   | 0.000248 | 4.04e+09 | 4.28 
	 float64 | (10000000,)  | 0.0148   | 6.74e+08 |      
	 float32 | (10000000,)  | 0.00783  | 1.28e+09 | 1.89 
	 float64 | (100000000,) | 0.142    | 7.06e+08 |      
	 float32 | (100000000,) | 0.0718   | 1.39e+09 | 1.97 


    exp(a)
	 dtype   | size         | s        | eval/s   | SP/DP
	---------+--------------+----------+----------+-------
	 float64 | (1000,)      | 1.47e-05 | 6.81e+07 |      
	 float32 | (1000,)      | 3.81e-06 | 2.62e+08 | 3.86 
	 float64 | (10000,)     | 0.000135 | 7.43e+07 |      
	 float32 | (10000,)     | 2.6e-05  | 3.85e+08 | 5.18 
	 float64 | (100000,)    | 0.00134  | 7.46e+07 |      
	 float32 | (100000,)    | 0.000248 | 4.04e+08 | 5.42 
	 float64 | (1000000,)   | 0.0135   | 7.43e+07 |      
	 float32 | (1000000,)   | 0.00249  | 4.02e+08 | 5.41 
	 float64 | (10000000,)  | 0.142    | 7.06e+07 |      
	 float32 | (10000000,)  | 0.0302   | 3.31e+08 | 4.69 
	 float64 | (100000000,) | 1.41     | 7.1e+07  |      
	 float32 | (100000000,) | 0.297    | 3.37e+08 | 4.75 


    sqrt(a)
	 dtype   | size         | s        | eval/s   | SP/DP
	---------+--------------+----------+----------+-------
	 float64 | (1000,)      | 3.08e-06 | 3.25e+08 |      
	 float32 | (1000,)      | 1.91e-06 | 5.24e+08 | 1.61 
	 float64 | (10000,)     | 1.94e-05 | 5.17e+08 |      
	 float32 | (10000,)     | 7.13e-06 | 1.4e+09  | 2.72 
	 float64 | (100000,)    | 0.000183 | 5.47e+08 |      
	 float32 | (100000,)    | 6.02e-05 | 1.66e+09 | 3.04 
	 float64 | (1000000,)   | 0.00183  | 5.46e+08 |      
	 float32 | (1000000,)   | 0.000587 | 1.7e+09  | 3.12 
	 float64 | (10000000,)  | 0.0257   | 3.89e+08 |      
	 float32 | (10000000,)  | 0.01     | 9.99e+08 | 2.57 
	 float64 | (100000000,) | 0.249    | 4.02e+08 |      
	 float32 | (100000000,) | 0.0939   | 1.07e+09 | 2.65 

## The heat equation revisited
### SP vs DP, blocked vs not blocked, Numpy array operations

    L1/7
	 description             | size      | s      | eval/s   | SP/DP | bl/nb
	-------------------------+-----------+--------+----------+-------+-------
	 float64_49x49           | 2883601   | 0.0783 | 3.68e+07 |       |      
	 float32_49x49           | 2883601   | 0.0702 | 4.11e+07 | 1.12  |      
	 float64_49x49_blocked   | 2883601   | 0.0577 | 5e+07    |       | 1.36 
	 float32_49x49_blocked   | 2883601   | 0.0622 | 4.63e+07 | 0.927 | 1.13 
	 float64_99x99           | 48034701  | 0.685  | 7.01e+07 |       |      
	 float32_99x99           | 48034701  | 0.493  | 9.74e+07 | 1.39  |      
	 float64_99x99_blocked   | 48034701  | 0.232  | 2.07e+08 |       | 2.95 
	 float32_99x99_blocked   | 48034701  | 0.25   | 1.92e+08 | 0.928 | 1.97 
	 float64_199x199         | 784139401 | 8.01   | 9.78e+07 |       |      
	 float32_199x199         | 784139401 | 5.76   | 1.36e+08 | 1.39  |      
	 float64_199x199_blocked | 784139401 | 1      | 7.84e+08 |       | 8.01 
	 float32_199x199_blocked | 784139401 | 1.07   | 7.31e+08 | 0.933 | 5.37 

### SP vs DP, blocked, Numpy array operations vs Numba.vectorize

	macbook pro m3
	repetitions = 5
	 description                   | size        | s      | eval/s   | SP/DP | np/u
	-------------------------------+-------------+--------+----------+-------+-------
	 float64_49x49_blocked         | 2883601     | 0.0593 | 4.86e+07 |       |      
	 float32_49x49_blocked         | 2883601     | 0.0602 | 4.79e+07 | 0.984 |      
	 float64_49x49_blocked_ufunc   | 2883601     | 0.0492 | 5.86e+07 |       | 1.21 
	 float32_49x49_blocked_ufunc   | 2883601     | 0.0542 | 5.32e+07 | 0.907 | 1.11 
	 float64_99x99_blocked         | 48034701    | 0.236  | 2.04e+08 |       |      
	 float32_99x99_blocked         | 48034701    | 0.259  | 1.85e+08 | 0.91  |      
	 float64_99x99_blocked_ufunc   | 48034701    | 0.211  | 2.27e+08 |       | 1.12 
	 float32_99x99_blocked_ufunc   | 48034701    | 0.224  | 2.15e+08 | 0.944 | 1.16 
	 float64_199x199_blocked       | 784139401   | 0.928  | 8.45e+08 |       |      
	 float32_199x199_blocked       | 784139401   | 1      | 7.84e+08 | 0.927 |      
	 float64_199x199_blocked_ufunc | 784139401   | 0.838  | 9.35e+08 |       | 1.11 
	 float32_199x199_blocked_ufunc | 784139401   | 0.929  | 8.44e+08 | 0.902 | 1.08 
	 float64_399x399_blocked       | 12672558801 | 3.89   | 3.26e+09 |       |      
	 float32_399x399_blocked       | 12672558801 | 4.23   | 3e+09    | 0.92  |      
	 float64_399x399_blocked_ufunc | 12672558801 | 3.47   | 3.65e+09 |       | 1.12 
	 float32_399x399_blocked_ufunc | 12672558801 | 3.85   | 3.29e+09 | 0.901 | 1.1  

	Vaughan
	repetitions = 5
	 description                   | size        | s     | eval/s   | SP/DP | np/u
	-------------------------------+-------------+-------+----------+-------+-------
	 float64_49x49_blocked         | 2883601     | 0.28  | 1.03e+07 |       |      
	 float32_49x49_blocked         | 2883601     | 0.285 | 1.01e+07 | 0.981 |      
	 float64_49x49_blocked_ufunc   | 2883601     | 0.244 | 1.18e+07 |       | 1.15 
	 float32_49x49_blocked_ufunc   | 2883601     | 0.273 | 1.06e+07 | 0.894 | 1.05 
	 float64_99x99_blocked         | 48034701    | 1.14  | 4.22e+07 |       |      
	 float32_99x99_blocked         | 48034701    | 1.17  | 4.12e+07 | 0.976 |      
	 float64_99x99_blocked_ufunc   | 48034701    | 1     | 4.8e+07  |       | 1.14 
	 float32_99x99_blocked_ufunc   | 48034701    | 1.11  | 4.31e+07 | 0.897 | 1.05 
	 float64_199x199_blocked       | 784139401   | 4.6   | 1.7e+08  |       |      
	 float32_199x199_blocked       | 784139401   | 4.73  | 1.66e+08 | 0.973 |      
	 float64_199x199_blocked_ufunc | 784139401   | 4.02  | 1.95e+08 |       | 1.15 
	 float32_199x199_blocked_ufunc | 784139401   | 4.51  | 1.74e+08 | 0.891 | 1.05 
	 float64_399x399_blocked       | 12672558801 | 18.7  | 6.79e+08 |       |      
	 float32_399x399_blocked       | 12672558801 | 18.9  | 6.69e+08 | 0.985 |      
	 float64_399x399_blocked_ufunc | 12672558801 | 16.2  | 7.81e+08 |       | 1.15 
	 float32_399x399_blocked_ufunc | 12672558801 | 18.2  | 6.97e+08 | 0.893 | 1.04 

### SP vs DP, blocked, Numpy array operations vs Numba.guvectorize

	macbook pro m3
	repetitions = 5
     description                    | size        | s      | eval/s   | SP/DP | np/gu
	--------------------------------+-------------+--------+----------+-------+-------
	 float64_49x49_blocked          | 2883601     | 0.0565 | 5.1e+07  |       |      
	 float32_49x49_blocked          | 2883601     | 0.0606 | 4.76e+07 | 0.933 |      
	 float64_49x49_blocked_gufunc   | 2883601     | 0.0474 | 6.09e+07 |       | 1.19 
	 float32_49x49_blocked_gufunc   | 2883601     | 0.0563 | 5.12e+07 | 0.841 | 1.08 
	 float64_99x99_blocked          | 48034701    | 0.231  | 2.08e+08 |       |      
	 float32_99x99_blocked          | 48034701    | 0.257  | 1.87e+08 | 0.898 |      
	 float64_99x99_blocked_gufunc   | 48034701    | 0.205  | 2.34e+08 |       | 1.13 
	 float32_99x99_blocked_gufunc   | 48034701    | 0.233  | 2.06e+08 | 0.878 | 1.1  
	 float64_199x199_blocked        | 784139401   | 0.942  | 8.32e+08 |       |      
	 float32_199x199_blocked        | 784139401   | 1.01   | 7.73e+08 | 0.929 |      
	 float64_199x199_blocked_gufunc | 784139401   | 0.8    | 9.8e+08  |       | 1.18 
	 float32_199x199_blocked_gufunc | 784139401   | 0.956  | 8.2e+08  | 0.837 | 1.06 
	 float64_399x399_blocked        | 12672558801 | 3.9    | 3.25e+09 |       |      
	 float32_399x399_blocked        | 12672558801 | 4.21   | 3.01e+09 | 0.928 |      
	 float64_399x399_blocked_gufunc | 12672558801 | 3.33   | 3.81e+09 |       | 1.17 
	 float32_399x399_blocked_gufunc | 12672558801 | 4      | 3.17e+09 | 0.832 | 1.05 

	Vaughan
	repetitions = 5
	 description                   | size        | s     | eval/s   | SP/DP | np/gu
	-------------------------------+-------------+-------+----------+-------+-------
	 float64_49x49_blocked         | 2883601     | 0.28  | 1.03e+07 |       |      
	 float32_49x49_blocked         | 2883601     | 0.285 | 1.01e+07 | 0.981 |      
	 float64_49x49_blocked_ufunc   | 2883601     | 0.244 | 1.18e+07 |       | 1.15 
	 float32_49x49_blocked_ufunc   | 2883601     | 0.273 | 1.06e+07 | 0.894 | 1.05 
	 float64_99x99_blocked         | 48034701    | 1.14  | 4.22e+07 |       |      
	 float32_99x99_blocked         | 48034701    | 1.17  | 4.12e+07 | 0.976 |      
	 float64_99x99_blocked_ufunc   | 48034701    | 1     | 4.8e+07  |       | 1.14 
	 float32_99x99_blocked_ufunc   | 48034701    | 1.11  | 4.31e+07 | 0.897 | 1.05 
	 float64_199x199_blocked       | 784139401   | 4.6   | 1.7e+08  |       |      
	 float32_199x199_blocked       | 784139401   | 4.73  | 1.66e+08 | 0.973 |      
	 float64_199x199_blocked_ufunc | 784139401   | 4.02  | 1.95e+08 |       | 1.15 
	 float32_199x199_blocked_ufunc | 784139401   | 4.51  | 1.74e+08 | 0.891 | 1.05 
	 float64_399x399_blocked       | 12672558801 | 18.7  | 6.79e+08 |       |      
	 float32_399x399_blocked       | 12672558801 | 18.9  | 6.69e+08 | 0.985 |      
	 float64_399x399_blocked_ufunc | 12672558801 | 16.2  | 7.81e+08 |       | 1.15 
	 float32_399x399_blocked_ufunc | 12672558801 | 18.2  | 6.97e+08 | 0.893 | 1.04 

### SP vs DP, blocked, Numpy array operations vs C++ implementation

	macbook-m3
	repetitions = 5
	 description                   | size        | s      | eval/s   | SP/DP | bl/nb
	-------------------------------+-------------+--------+----------+-------+-------
	 float64_49x49_blocked         | 2883601     | 0.056  | 5.15e+07 |       |      
	 float32_49x49_blocked         | 2883601     | 0.0602 | 4.79e+07 | 0.931 |      
	 float64_49x49_blocked_afunc   | 2883601     | 0.0457 | 6.31e+07 |       | 1.23 
	 float32_49x49_blocked_afunc   | 2883601     | 0.0499 | 5.78e+07 | 0.916 | 1.21 
	 float64_99x99_blocked         | 48034701    | 0.228  | 2.11e+08 |       |      
	 float32_99x99_blocked         | 48034701    | 0.255  | 1.88e+08 | 0.893 |      
	 float64_99x99_blocked_afunc   | 48034701    | 0.195  | 2.47e+08 |       | 1.17 
	 float32_99x99_blocked_afunc   | 48034701    | 0.213  | 2.26e+08 | 0.916 | 1.2  
	 float64_199x199_blocked       | 784139401   | 0.932  | 8.41e+08 |       |      
	 float32_199x199_blocked       | 784139401   | 0.999  | 7.85e+08 | 0.933 |      
	 float64_199x199_blocked_afunc | 784139401   | 0.759  | 1.03e+09 |       | 1.23 
	 float32_199x199_blocked_afunc | 784139401   | 0.872  | 8.99e+08 | 0.87  | 1.15 
	 float64_399x399_blocked       | 12672558801 | 3.91   | 3.24e+09 |       |      
	 float32_399x399_blocked       | 12672558801 | 4.37   | 2.9e+09  | 0.894 |      
	 float64_399x399_blocked_afunc | 12672558801 | 3.44   | 3.69e+09 |       | 1.14 
	 float32_399x399_blocked_afunc | 12672558801 | 3.68   | 3.44e+09 | 0.935 | 1.19 

	vaughan
	repetitions = 5
	 description                   | size        | s     | eval/s   | SP/DP | bl/nb
	-------------------------------+-------------+-------+----------+-------+-------
	 float64_49x49_blocked         | 2883601     | 0.272 | 1.06e+07 |       |      
	 float32_49x49_blocked         | 2883601     | 0.283 | 1.02e+07 | 0.963 |      
	 float64_49x49_blocked_afunc   | 2883601     | 0.225 | 1.28e+07 |       | 1.21 
	 float32_49x49_blocked_afunc   | 2883601     | 0.244 | 1.18e+07 | 0.923 | 1.16 
	 float64_99x99_blocked         | 48034701    | 1.12  | 4.28e+07 |       |      
	 float32_99x99_blocked         | 48034701    | 1.16  | 4.13e+07 | 0.965 |      
	 float64_99x99_blocked_afunc   | 48034701    | 0.92  | 5.22e+07 |       | 1.22 
	 float32_99x99_blocked_afunc   | 48034701    | 0.989 | 4.86e+07 | 0.931 | 1.18 
	 float64_199x199_blocked       | 784139401   | 4.53  | 1.73e+08 |       |      
	 float32_199x199_blocked       | 784139401   | 4.68  | 1.67e+08 | 0.966 |      
	 float64_199x199_blocked_afunc | 784139401   | 3.7   | 2.12e+08 |       | 1.22 
	 float32_199x199_blocked_afunc | 784139401   | 4.01  | 1.96e+08 | 0.924 | 1.17 
	 float64_399x399_blocked       | 12672558801 | 18.3  | 6.94e+08 |       |      
	 float32_399x399_blocked       | 12672558801 | 18.8  | 6.73e+08 | 0.969 |      
	 float64_399x399_blocked_afunc | 12672558801 | 14.9  | 8.48e+08 |       | 1.22 
	 float32_399x399_blocked_afunc | 12672558801 | 16    | 7.9e+08  | 0.932 | 1.17 

## conclusion

Nothing helps and i have no clue...