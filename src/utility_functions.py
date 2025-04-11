import numpy as np
import pandas as pd
import math

def df_to_cppvector(df):
    surveytmp = []

    for col in df.columns:
        col_data = df[col]
        coltmp = []
        for val in col_data:
            if pd.isnull(val):
                coltmp.append(np.nan)
            elif isinstance(val, str):
                try:
                    coltmp.append(float(val))
                except ValueError:
                    coltmp.append(np.nan)
            else:
                coltmp.append(float(val))
        surveytmp.append(coltmp)

    # Transpose so that each row is a user
    stmp = list(map(list, zip(*surveytmp)))
    return stmp


def normalise_columns(s):
    s = np.array(s)
    colmin = np.nanmin(s, axis=0)
    colmax = np.nanmax(s, axis=0)
    m = 2 / (colmax - colmin)
    b = -(colmax + colmin) / (colmax - colmin)
    normed = m * s + b
    return normed.tolist()


def cppvector_to_df(graph_obj, c):
    rows = []
    for u, neighbors in graph_obj['network'].items():
        for edge in neighbors:
            if u < edge['u']:
                weight = round(edge['w'], 4)
                if c == 0:
                    weight += 1.0
                rows.append({'u': u + 1, 'v': edge['u'] + 1, 'weight': weight})
    return pd.DataFrame(rows)
