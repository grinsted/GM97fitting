import numpy as np
from numba import njit


@njit
def find_span(t, k, x):
    n = len(t) - k - 1

    if x >= t[n]:
        return n - 1
    if x <= t[k]:
        return k

    low = k
    high = n
    mid = (low + high) // 2

    while x < t[mid] or x >= t[mid + 1]:
        if x < t[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


@njit
def de_boor_single(x, t, c, k):
    i = find_span(t, k, x)

    d = np.empty(k + 1)
    for j in range(k + 1):
        d[j] = c[i - k + j]

    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left = t[i - k + j]
            right = t[i + 1 + j - r]

            if right - left == 0.0:
                alpha = 0.0
            else:
                alpha = (x - left) / (right - left)

            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[k]


@njit
def spline_eval(x_vals, t, c, k):
    n = len(x_vals)
    out = np.empty(n)

    for idx in range(n):
        out[idx] = de_boor_single(x_vals[idx], t, c, k)

    return out
