import pandas as pd
import numpy as np
import random
import warnings

def clean_minority(m):
    if m < 0.0 or m > 1.0:
        warnings.warn("'minority' must be a fraction between 0 and 0.5, setting to 0.5")
        return 0.5
    elif m > 0.5:
        warnings.warn("'minority' must be between 0 and 0.5, taking 1 - minority")
        return 1 - m
    return m

def clean_polarisation(p):
    if p < 0:
        warnings.warn("'polarisation' has been rounded up to 0")
        return 0
    elif p > 1:
        warnings.warn("'polarisation' has been rounded down to 1")
        return 1
    return p

def clean_correlation(c):
    if c < 0:
        warnings.warn("'correlation' has been rounded up to 0")
        return 0
    elif c > 1:
        warnings.warn("'correlation' has been rounded down to 1")
        return 1
    return c

def clean_scale(s):
    if s < 2:
        warnings.warn("'scale' must be integer and greater than two, setting to 5")
        return 5
    return s

def make_synthetic_data(nrow, 
                        ncol,
                        minority=0.5,
                        polarisation=0,
                        correlation=0.85,
                        scale=10):

    minority = clean_minority(minority)
    polarisation = clean_polarisation(polarisation)
    correlation = clean_correlation(correlation)
    scale = clean_scale(scale)

    data = pd.DataFrame(np.full((nrow, ncol + 1), np.nan))

    avg = scale / 2 + (1 - scale / 2) * polarisation

    for i in range(nrow):
        # group column
        if i < int(minority * nrow):
            group = '0' if random.random() < correlation else '1'
        else:
            group = '1' if random.random() < correlation else '0'

        data.iloc[i, 0] = group

        for j in range(ncol):
            # avgflag logic
            if i < int(minority * nrow):
                avgflag = 1 if j < 0.5 * ncol else 0
            else:
                avgflag = 0 if j < 0.5 * ncol else 1

            val = np.random.poisson(avg)
            while val < 1 or val > scale:
                val = np.random.poisson(avg)

            if avgflag == 1:
                val = scale + 1 - val

            data.iloc[i, j + 1] = val

            # To introduce random NAs (optional):
            # if random.random() < 0.15:
            #     data.iloc[i, j + 1] = np.nan

    # Column names: group, item_1, item_2, ...
    columns = ['group'] + [f'item_{i+1}' for i in range(ncol)]
    data.columns = columns

    return data
