import numpy as np

def integral(a0, v0, x0, j, t):

    a = a0 + j * t
    v = v0 + a0 * t + 0.5 * j * t**2
    x = x0 + v0 * t + 0.5 * a0 * t**2 + 1.0 / 6.0 * j * t**3

    p = np.array([[a, v, x]]).T

    return p
