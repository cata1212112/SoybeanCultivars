import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


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


def get_kfold(dataframe):
    kf = KFold(n_splits=10, random_state=7, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(dataframe)):
        yield train_index, test_index
