# -*- coding: utf-8 -*-

"""
## Python (sub)module heat
"""

import dpvssp.heat.process  as process
import dpvssp.heat.material as material
import dpvssp.heat.geometry as geometry

from numpy import float64, float32, pi
import numpy as np



def compute_Txy(t, x, y=0.0, T0=300.
                , experiment: int = 0
                , report_n_clipped=False
                , ufunc=''
                , blocked=False
                ):
    """Compute the temperature rise at the origin for the time/pulse location
    table `(t,x,y)`.

    Args:
        t: numpy array with the times of the pulses (monotonously increasing).
            Size = n_pulses+1.
        x: numpy array with the x-coordinate of the pulse locations.
            Size = n_pulses.
        y: numpy array with the y-coordinate of the pulse locations
            Size = n_pulses.
        T0: material temperature at t=t[0]=0.
        experiment: process parameters (Eth, omega) are taken from this
            experiment.

    Returns:
       T: array, T[i] is the temperature at time t[i+1] due to all
            pulses applied before t[i+1]. So T[0] is the temperature at t[1],
            and so on.

    This algorithm computes the contribution of pulse 1 to all times, then the
    contribution of pulse 2, ... After each pulse the material parameters are
    re-interpolated. This approach also allows for array computations, where
    the arrays are arrays over quantities at different times but at the
    same pulse location. At the same time the approach is physically more
    correct.
    """
    # We assume that we only want to know the temperature at the pulse times
    # the algorithm can be adapted to compute at other times as well.
    if t.dtype.type is np.float64:
        float_type = np.float64
    elif t.dtype.type is np.float32:
        float_type = np.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {t.dtype}")

    if not isinstance(y, np.ndarray):
        y = float_type(y)

    if not isinstance(T0, np.ndarray):
        T0 = float_type(T0)

    if not  x.dtype is t.dtype \
    or not  y.dtype is t.dtype \
    or not T0.dtype is t.dtype:
        raise ValueError(f'Args x, y and T0 must have the same dtype as t ({t.dtype})')

    xc = float_type(0.)
    yc = float_type(0.)

    # location where we want to know the temperature

    # Process parameters
    Eth, omega0 = process.get_Eth_omega0(experiment=experiment)
    # derived quantities that remain constant over time
    Ethpi32rho = float_type(2 * Eth / (pi * np.sqrt(pi) * material.rho))
    w2         = float_type(0.5 * omega0 ** 2)

    n_times = len(t)
    n_pulses = len(x)
    ntot_clipped = 0
    ntot = 0

    # if dtype not in (float, np.float64):
    pola, polc = material.get_material_polynomials(float_type)

    # Allocate work arrays
    _a      = np.empty_like(x)
    _c      = np.empty_like(x)
    _dt     = np.empty_like(x)
    _a4dt   = np.empty_like(x)
    _a4dtw2 = np.empty_like(x)
    _Tclipd = np.empty_like(x)

    t1 = t[1:]  # len(t1) == len(x), t has one item more that

    T = T0 * np.ones_like(x)

    if blocked:
        # Compute the time slices in blocks that optimize cache utilization
        L1cache = 32 * 1024
        blocksize = L1cache // t.dtype.itemsize // 7
        nblocks = int(np.ceil(n_pulses / blocksize))
    else:
        # only one block
        blocksize = n_pulses
        nblocks = 1

    for iblock in range(nblocks):
        blockstart = iblock * blocksize
        blockstop = min(blockstart + blocksize, n_pulses)

        for i in range(blockstart, blockstop):
            # slice of of times pulse i contributes to
            slice = np.s_[i:blockstop:]

            # Clip the temperature
            if report_n_clipped:
                n_clipped, n = (T[slice] > 1073).sum(), len(T[slice])
                print(f"pulses>={i}, t={t[i + 1]}: n_clipped = {n_clipped}/{n} = {100 * n_clipped / n:5.1f}%")
                ntot_clipped += n_clipped
                ntot += n
            np.clip(T[slice], 0, 1073, out=_Tclipd[slice])

            # Interpolate the material parameters from the clipped temperatures
            _a[slice] = pola(_Tclipd[slice])
            _c[slice] = polc(_Tclipd[slice])

            _dt[slice] = t1[slice] - t[i]
            xi = x[i]
            yi = y[i] if isinstance(y, np.ndarray) else y
            #########################################
            # md2 = - ( (xi-xc)**2 + (yi-yc)**2 )   #
            # GOTCHA! float32**2 returns a float64. #
            # That was unexpected!                  #
            #########################################
            md2 = - ((xi - xc) * (xi - xc) + (yi - yc) * (yi - yc))

            # Compute temperature rise of pulse i at all subsequent times
            if not ufunc:
                # simple numpy array operations (involving temporary arrays.)
                _a4dt[slice] = (4 * _a[slice]) * t1[slice]
                _a4dtw2[slice] = _a4dt[slice] + w2

                Trise = ((Ethpi32rho / _c[slice])
                         / (np.sqrt(_a4dt[slice]) * _a4dtw2[slice])
                         ) * np.exp(md2 / _a4dtw2[slice]
                                    # - (z-z0)**2 / alpha4tx # assuming z=z0
                                    )
                T[slice] += Trise

            # elif ufunc == 'ufunc':
            #     # use numpy universal func by numba
            #     Trise = ufunc_Trise(Ethpi32rho, w2
            #                         , _a[slice], _c[slice]
            #                         , md2
            #                         , t1[slice]
            #                         )
            #     T[slice] += Trise
            #
            # elif ufunc == 'afunc':
            #     if dtype in (float, np.float64):
            #         afuncDP(Ethpi32rho, w2, _a[slice], _c[slice], md2, t1[slice]
            #                 , T[slice]
            #                 )
            #     elif dtype is np.float32:
            #         afuncSP(Ethpi32rho, w2, _a[slice], _c[slice], md2, t1[slice]
            #                 , T[slice]
            #                 )
            #     else:
            #         raise NotImplementedError(f'Unknown dtype: {dtype}')

            else:
                raise NotImplementedError(f'ufunc `{ufunc}` not supported')


    if report_n_clipped:
        print(f"ntot_clipped = {ntot_clipped}/{ntot} = {100 * ntot_clipped / ntot:5.1f}%\n")
    return T


if __name__ == '__main__':
    experiment = 0
    t,x,y = generate_square(1,experiment=experiment, dtype=float64)
    T = compute_Txy(t, x ,y, T0=300, experiment=experiment)
    print('-*# finished #*-')
