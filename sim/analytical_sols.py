#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.special import expi

alpha_t = 3.484e-4
alpha_p = 19.03
ct = 200e-6


def source_line_solution(r, t, q, viso, perm, por, base_press, dk):
    """Source-line solution in a radial-infinite homogeneous reservoir.

    Parameters
    ----------
    r : number
        Investigation radius
    t : number
        Time in days
    reservoir_params : dict
        Reservoir parameters, such as:
        Q - Oil flow ratio
        VISO - Oil viscosity
        PERMI - Permeability in the I direction
        POR - Porosity
        REFPRES - Reference Bottom-hole pressure
        DK - thickness of each reservoir slice

    Returns
    -------
    A number with the estimated pressure value at radius 'r' and time 't' for a
    reservoir with the given parameters.
    """
    h = dk[0] * dk[1]
    a = (alpha_p * q * viso) / (perm * h)
    b = -0.5 * expi(-(por * viso * ct * r * r) / (4 * alpha_t * perm * t))
    return a * b


def investigation_radius_solution(t, perm, por, viso):
    """Investigation radius solution in a radial-infinite homogeneous
    reservoir.

    Parameters
    ----------
    t : number
        Time in days
    perm : number
        Reservoir permeability in the I direction.
    por : number
        Reservoir porosity.
    viso : number
        Oil viscosity.

    Returns
    -------
    A number with the estimated pressure investigation radius at time 't' for a
    reservoir with the given parameters.
    """
    tD = (3.484e-4 * perm * t) / (np.exp(0.5778) * por * viso * 200e-6)
    return 1.5 * np.sqrt(tD)


def test_source_line():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sim import calc_log_derivative

    por = 0.3
    viso = 1
    q = 500
    base_press = 300
    dk = (20, 100)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlabel('Time(days)')
    ax[0].set_ylabel('dP()')
    time = np.arange(0.1, 5, 0.1)
    for permi in np.arange(500, 5500, 500):
        for r in np.arange(50, 550, 50):
            dp = 300-source_line_solution(r, time, q, viso, permi, por,
                                          base_press, dk)
            ddp = calc_log_derivative(pd.Series(data=dp, index=time))
            ax[0].plot(time, dp)
            ddp.plot(ax=ax[1])

    plt.show()
    plt.close()


def test_investigation_radius():
    import matplotlib.pyplot as plt
    por = 0.3
    viso = 2

    fig, ax = plt.subplots()
    ax.set_xlabel('Time(days)')
    ax.set_ylabel('Radius(m)')
    time = np.arange(0.1, 5, 0.1)
    for permi in [500, 1500, 3000, 5000]:
        r = []
        for t in time:
            r.append(investigation_radius_solution(t, permi, por, viso))
        ax.plot(time, r, label='PERMI {}'.format(permi))

    ax.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    test_source_line()
