# coding: utf-8

from enum import IntEnum, auto

import numpy as np
import pandas as pd


class DerivativeType(IntEnum):
    POINT_2 = auto()
    POINT_3 = auto()


def calc_log_derivative(dP, algorithm=DerivativeType.POINT_3):
    """Calculate the log derivative of a time-series.

    Parameters
    ----------
    dP : pandas.Series
        The time-series itself with the time-steps as indices.
    algorithm : DerivativeType, optional
        The algorithm to use when calculating the derivative. Default value is
        POINT_3, meaning that the 3-point algorithm is used instead of the
        usual 2-point algorithm.

    Returns
    -------
    pandas.Series
        The time-series derivative with `log(timesteps)` as its indices.
    """
    T = dP.index
    deltap = None

    if algorithm == DerivativeType.POINT_3:
        deltap = pd.Series(data=np.zeros(len(dP) - 2), index=np.log(T[1:-1]))
        for i in range(1, len(T) - 1):
            d1 = np.log(T[i + 1] / T[i])
            c1 = np.log(T[i] / T[i - 1])
            c2 = np.log(T[i + 1] / T[i - 1])
            m1 = c1 / c2
            m2 = d1 / c2
            term1 = (dP.iloc[i + 1] - dP.iloc[i]) / d1 * m1
            term2 = (dP.iloc[i] - dP.iloc[i - 1]) / c2 * m2
            deltap.iloc[i - 1] = term1 + term2
    else:
        deltap = pd.Series(data=np.zeros(len(dP) - 1), index=np.log(T[1:]))
        for i in range(1, len(T)):
            deltap.iloc[i - 1] = (dP.iloc[i] - dP.iloc[i - 1]) / np.log(T[i] / T[i - 1])

    return deltap


def calc_derivative(dP):
    """Calculates the derivative of a given time-series

    Parameters
    ----------
    dP : pandas.Series
        The time-series itself with the time-steps as indices.

    Returns
    -------
    pandas.Series
        The time-series derivative.
    """
    T = dP.index
    deltap = pd.Series(data=np.zeros(len(dP) - 1), index=np.log(T[1:]))
    for i in range(1, len(T)):
        deltap.iloc[i - 1] = (dP.iloc[i] - dP.iloc[i - 1]) / (T[i] - T[i - 1])

    return deltap
