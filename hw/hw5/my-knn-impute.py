import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers

def distance_matric(data):
    n_variables = data.shape[1]
    n_observations = data.shape[0]
    
    is_numeric = [all(isinstance(n, numbers.Number) for n in data.iloc[:, i]) for i, x in enumerate(data)]
    n_numeric_var = sum(is_numeric)
    n_categorical_var = n_variables - n_numeric_var
    data_numeric = data.iloc[:, is_numeric]
    data_numeric = (data_numeric - data_numeric.mean()) / (data_numeric.max() - data_numeric.min())
    data_categorical = data.iloc[:, [not x for x in is_numeric]]

    data_numeric.fillna(data_numeric.mean(), inplace=True)
    for x in data_categorical:
        data_categorical[x].fillna(data_categorical[x].mode()[0], inplace=True)

    data_categorical = pd.DataFrame([pd.factorize(data_categorical[x])[0] for x in data_categorical]).T
    
    result_numeric = cdist(data_numeric, data_numeric, metric='euclidean')
    result_categorical = cdist(data_categorical, data_categorical, metric='hamming')
    result_matrix = np.array([[1.0*(result_numeric[i, j] * n_numeric_var + result_categorical[i, j] *
                               n_categorical_var) / n_variables for j in range(n_observations)] for i in range(n_observations)])

    np.fill_diagonal(result_matrix, np.nan)

    return pd.DataFrame(result_matrix)



def knn_impute(column, dataframe, k, agg_fn="mean"):

    column = pd.DataFrame(column)
    dataframe = pd.DataFrame(dataframe)

    distances = distance_matric(dataframe)
    if distances is None:
        return None

    for i, value in enumerate(column.iloc[:, 0]):
        if pd.isnull(value):
            order = distances.iloc[i,:].values.argsort()[:k]
            closest_to_target = column.iloc[order, :]
            if agg_fn == "mean":
                column.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            elif agg_fn == "median":
                column.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            elif agg_fn == 'mode':
                column.iloc[i] = stats.mode(closest_to_target, nan_policy='omit')[0][0]

    return column

    
''' To use it
titanic_train['survived'] = knn_impute(column=titanic_train['survived'], dataframe=titanic_train.drop(['survived'], 1),
                                    agg_fn="mode", k=10)
'''
    




    
    