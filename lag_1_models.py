from cgi import test
from select import select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import pickle
from sklearn.model_selection import train_test_split
import re
from arch.unitroot.cointegration import engle_granger


from FRUFS import FRUFS
import matplotlib.pyplot as plt
import optuna
import joblib, gc
import seaborn as sns

from sklearn.datasets import make_regression
from scipy.stats import pearsonr
from tqdm.notebook import trange, tqdm
from FRUFS import FRUFS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import meanabs
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import grangercausalitytests
import pickle
from sklearn.model_selection import train_test_split
import re
import logging
import plotly.express as px
import math
import statistics
from sklearn.model_selection import TimeSeriesSplit
from Sun_Model_Class import Sun_Model
import pmdarima as pmd
from my_metrics import rae, rrse

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from imports.skfeature.function.similarity_based import fisher_score
from lag_functions import lag_n_times






def stationarise(df, target, test_size,):

    lag_1df =  lag_n_times(df, 1, target=target, selected_f_names=None)

    
    feature_df = lag_1df.loc[:, ~lag_1df.columns.isin([target, "Date"])]
    target_df = lag_1df.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=test_size, shuffle=False)
    
    # Resetting index for later modelling purposes
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Dataframe to perform ADF test
    staionarity_df = pd.concat([y_train, X_train], axis=1)
    features = list(staionarity_df.columns)
    
    # Coppying dataframes for stationarity and back-transformation purposes
    yx_train_original = staionarity_df.copy()
    y_original = y_train.copy()
    y_logged = np.log(y_original)
    orders_of_integ = {}
    const_counters = {}

    for feature in features:
        result = adfuller(staionarity_df[feature], autolag="t-stat", regression="c")
        counter = 0

        while result[1] >= 0.01:
            staionarity_df[feature] = staionarity_df[feature] - staionarity_df[feature].shift(1)
            counter += 1
            #dropna(inplace=False) because it drops one observation for each feature
            result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat", regression="c")
        print(f'Order of integration for feature "{feature}" is {counter}')
        orders_of_integ[feature] = counter

    staionarity_df.dropna(inplace=True)
    staionarity_df.reset_index(drop=True, inplace=True)
    y_train = staionarity_df[target]
    X_train = staionarity_df[[n for n in list(staionarity_df.columns) if n != target]]








    # Stationarising test data
    stationarity_df_test = pd.concat([y_test, X_test], axis=1)
    features = list(stationarity_df_test.columns)
    for feature in features:
        # Continue if the feature was found to be stationary without tranformation
        if orders_of_integ[feature] == 0:
            continue
        else:
            order = orders_of_integ[feature]
            integr_list = list(range(order, order+1))
            # Difference o times as with the train data
            for o in integr_list:
                stationarity_df_test[feature] = stationarity_df_test[feature].diff()
    stationarity_df_test.dropna(inplace=True)
    stationarity_df_test.reset_index(drop=True, inplace=True)
    y_test = stationarity_df_test[target]
    X_test = stationarity_df_test[[n for n in list(stationarity_df_test.columns) if n != target]]

    return X_train, X_test, y_train, y_test




# df_aapl = pd.read_csv("df_aaple.csv")
# # Truncating the dataw
# aapl_long = df_aapl.iloc[:,:22]


# X_train, X_test, y_train, y_test = stationarise(aapl_long, "Close", 0.2)

# model = LinearRegression().fit(X_train, y_train)

# #Calculating train set error
# pred_in = model.predict(X_train)
# MAE_train = mean_absolute_error(y_train, pred_in)



# #Calculating CV set error
# tscv = TimeSeriesSplit(n_splits=5)
# MAE_CV_list = []
# RAE_CV_list = []
# RAE_in_list = []
# for train_index, test_index in tscv.split(X_train):
#     X_CV_train, X_CV_test = X_train.iloc[train_index], X_train.iloc[test_index]
#     y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]

#     model_CV = LinearRegression().fit(X_CV_train, y_CV_train)
#     y_CV_pred = model_CV.predict(X_CV_test)
#     MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
#     MAE_CV_list.append(MAE_CV)
#     RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
#     RAE_CV_list.append(RAE_CV)

# MAE_CV = statistics.mean(MAE_CV_list)







# #Calculating the error on the test set
# pred_out = model.predict(X_test)
# MAE_test = mean_absolute_error(y_test, pred_out)

# MAE = [MAE_train, MAE_CV, MAE_test]

# print(MAE)