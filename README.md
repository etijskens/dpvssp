# Python package dpvssp - experimenting with SP vs DP numpy arrays

The expectation that single precision arithmetic is in general twice as fast as double precision stems from the fact that

* single precision data occupies half the amount of bytes in mainn memory, cache and registers,
* vector registers therefore containt twice as much data items,
* the bandwith in terms of data items per second is twice as big.

Yet, in a previous project we struggled with achieving better performance in single precision (SP) than in double precision (DP). The setting was a Python/ program evaluating a function over a large number of points, stored as numpy arrays and using the numba.vectorize decorator.

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



https://stackoverflow.com/questions/77028828/why-are-numpy-operations-with-float32-significantly-faster-than-float64