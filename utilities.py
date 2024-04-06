import pandas as pd
import numpy as np


def clip_outliers(dataframe):
    float_columns = dataframe.select_dtypes(include='float64')

    for col in float_columns:
        q1 = np.percentile(dataframe[col], 25)
        q3 = np.percentile(dataframe[col], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        dataframe[col] = dataframe[col].clip(lower_bound, upper_bound)

    return dataframe
