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

from FRUFS import FRUFS
from sklearn.model_selection import train_test_split
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

from lag_functions import lag_n_times, lag_n_times_evoML
import time



def RFE_FS(df, target, test_size, n_features, n_features_2, max_lag):

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

   

    # Creating datasets with additional 19 lags for all features (to be used later)
    Xs = [X_train, X_test]
    split = ["train", "test"]
    Xs_all_lags = {}
    counter = 0
    for X in Xs:
        var_names = list(X.columns)

        names_without_L = []
        X_all_lags = []
        for name in var_names:
            name_without_L = name.split(".L")[0]
            names_without_L.append(name_without_L)

            columns = []
            for i in list(range(2, max_lag+1+1)):
                columns.append(name_without_L + ".L"+str(i))

            feature_n_df = pd.DataFrame(columns=columns)
            for i in list(range(max_lag)):
                feature_n_df[columns[i]] = X[name].shift(i+1)

            feature_n_df = pd.concat([X[name], feature_n_df], axis=1)
            X_all_lags.append(feature_n_df)

        X_all_lags = pd.concat(X_all_lags, axis=1)
        X_all_lags = X_all_lags.iloc[max_lag:]
        X_all_lags.reset_index(drop=True, inplace=True)
        Xs_all_lags[split[counter]] = X_all_lags
        counter += 1

    X_train_all_lags = Xs_all_lags["train"]
    X_test_all_lags = Xs_all_lags["test"]










    # Recording start time
    start = time.time()
   
    #Selecting the features ans saving them
    estimator = LinearSVR()
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector_fit = selector.fit(X_train, y_train)
    selected_names = list(selector_fit.get_feature_names_out())
    X_selected = selector_fit.transform(X_train)
    X_selected = pd.DataFrame(X_selected, columns=selected_names)



    #Fitting the model on the selected features
    model = LinearRegression().fit(X_selected, y_train)




    #Calculating train set error
    pred_in = model.predict(X_selected)
    MAE_train = mean_absolute_error(y_train, pred_in)





    #Calculating CV set error
    tscv = TimeSeriesSplit(n_splits=5)
    MAE_CV_list = []
    RAE_CV_list = []
    RAE_in_list = []
    for train_index, test_index in tscv.split(y_train):
        X_CV_train, X_CV_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]

        model_CV = LinearRegression().fit(X_CV_train, y_CV_train)
        y_CV_pred = model_CV.predict(X_CV_test)
        MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
        MAE_CV_list.append(MAE_CV)
        RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
        RAE_CV_list.append(RAE_CV)
    
    MAE_CV = statistics.mean(MAE_CV_list)


    




    #Calculating the error on the test set
    X_selected_test = X_test[selected_names]
    pred_out = model.predict(X_selected_test)
    MAE_test = mean_absolute_error(y_test, pred_out)






    #Deleting the lags of the not selected variables
    #List of selected variables without the Lag notation at the end
    selected_names_without_L = []
    for name in selected_names:
        name_without_L = name.split(".L")[0]
        selected_names_without_L.append(name_without_L)




    Xs = [X_train_all_lags, X_test_all_lags]
    split = ["train", "test"]
    Xs_all_lags = {}
    counter = 0
    for X in Xs:

        X_all_lags_names = list(X.columns) 
        total_n_names = len(X_all_lags_names)
        n_unique_vars = int(total_n_names/(max_lag+1))
        n_unique_vars = list(range(1, n_unique_vars+1))
        for n in n_unique_vars:
            start_index = n*(max_lag+1) - (max_lag+1)
            last_index = n*(max_lag+1)
            unique_var_lags = X_all_lags_names[start_index:last_index]
            unique_var_name = unique_var_lags[0].split(".L")[0]
            if unique_var_name not in selected_names_without_L:
                for lag in unique_var_lags:
                        X.pop(lag)
            else:
                continue

        Xs_all_lags[split[counter]] = X
        counter += 1
    
    X_train_selected_all_lags = Xs_all_lags["train"]
    X_test_selected_all_lags = Xs_all_lags["test"]


    #Truncating y_train so that its length corresponds to that of Xs
    y_train=y_train.iloc[max_lag:]
    y_train.reset_index(drop=True, inplace=True)








    #Applying feature selection to all the lags of the unique selected variables    
    selector = RFE(estimator, n_features_to_select=n_features_2, step=1)
    selector_fit = selector.fit(X_train_selected_all_lags, y_train)
    selected_names_new = list(selector_fit.get_feature_names_out())
    X_selected = selector_fit.transform(X_train_selected_all_lags)
    X_selected = pd.DataFrame(X_selected, columns=selected_names_new)

    # Recording end time
    end = time.time()
    time_to_run = end - start

    #Fitting the model on the selected lags
    model = LinearRegression().fit(X_selected, y_train)
    #Calculating the train set error
    pred_in = model.predict(X_selected)
    MAE_train_all = mean_absolute_error(y_train, pred_in)





    #Calculating the CV set error
    tscv = TimeSeriesSplit(n_splits=5)
    MAE_CV_list = []
    RAE_CV_list = []
    RAE_in_list = []
    for train_index, test_index in tscv.split(y_train):
        X_CV_train, X_CV_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]

        model_CV = LinearRegression().fit(X_CV_train, y_CV_train)
        y_CV_pred = model_CV.predict(X_CV_test)
        MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
        MAE_CV_list.append(MAE_CV)
        RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
        RAE_CV_list.append(RAE_CV)
    
    MAE_CV_all = statistics.mean(MAE_CV_list)









    #Deleting the not selected lags
    X_test_selected_all_lags = X_test_selected_all_lags[selected_names_new]
    #Truncating the length of y so that it corresponds to that of X
    y_test = y_test.iloc[max_lag:]
    y_test.reset_index(drop=True, inplace=True)




    #Calculating the train set error
    pred_out = model.predict(X_test_selected_all_lags)
    MAE_test_all = mean_absolute_error(y_test, pred_out)


    # Saving the selected subsets (evoML requires special formatiing, so it is saved seperately)
    selected_df_evoML = lag_n_times_evoML(df, max_lag+1, target=target, selected_f_names=selected_names_new)
    selected_df = lag_n_times(df, max_lag+1, target=target, selected_f_names=selected_names_new)



    return selected_names, selected_names_new, [MAE_train, MAE_CV, MAE_test], [MAE_train_all, MAE_CV_all, MAE_test_all], selected_df_evoML, selected_df, time_to_run




def BE(df, target, test_size, n_features, n_features_2, max_lag):

    # Recording start time
    start = time.time()


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

   

    # Creating datasets with additional 19 lags for all features (to be used later)
    Xs = [X_train, X_test]
    split = ["train", "test"]
    Xs_all_lags = {}
    counter = 0
    for X in Xs:
        var_names = list(X.columns)

        names_without_L = []
        X_all_lags = []
        for name in var_names:
            name_without_L = name.split(".L")[0]
            names_without_L.append(name_without_L)

            columns = []
            for i in list(range(2, max_lag+1+1)):
                columns.append(name_without_L + ".L"+str(i))

            feature_n_df = pd.DataFrame(columns=columns)
            for i in list(range(max_lag)):
                feature_n_df[columns[i]] = X[name].shift(i+1)

            feature_n_df = pd.concat([X[name], feature_n_df], axis=1)
            X_all_lags.append(feature_n_df)

        X_all_lags = pd.concat(X_all_lags, axis=1)
        X_all_lags = X_all_lags.iloc[max_lag:]
        X_all_lags.reset_index(drop=True, inplace=True)
        Xs_all_lags[split[counter]] = X_all_lags
        counter += 1

    X_train_all_lags = Xs_all_lags["train"]
    X_test_all_lags = Xs_all_lags["test"]





    #Fitting the model on all features
    model = AutoReg(y_train, lags=0, exog=X_train).fit()

   
    # Using backward elimination to drop insignificant features
    # Defining critiacl p-value determining whether a feture is to be dropped
    critical_p_value = 0.05
    # Finding p-value of the lesat siginificant feature
    max_p_value = max(model.pvalues)
    # Defining const_dropped to know whether we run Autoreg with or w/o const
    const_dropped = False
    while max_p_value >= critical_p_value:
        # Column name of the least significant feature
        least_sig_var = list(model.params[np.where(model.pvalues == max_p_value)[0]].index)[0]
        # If least_sig_var is the constant we run Autoreg without it
        if least_sig_var == "const":
            model = AutoReg(y_train, lags=0, exog=X_train, trend="n").fit()
            const_dropped = True

        else:
            # Dropping the least_sig_var from the df
            X_train.pop(least_sig_var)
            # If const has been dropped, we run Autoreg w/o it
            if const_dropped:
                try:
                    model = AutoReg(y_train, lags=0, exog=X_train, trend="n").fit()
                except ValueError:
                    print("\n\n\n\nNo coefficients appear to be significant in the estimated model.\n\n\n\n")
            else:
                model = AutoReg(y_train, lags=0, exog=X_train).fit()

        # At the end of each iteration we find the new highest p-value        
        max_p_value = max(model.pvalues)

    # Defining list of all significant variables (except for const - because it is not in the data)
    selected_names = [n for n in list(model.params.index) if n!= "const"]
    X_selected = X_train
    if const_dropped:
        model = LinearRegression(fit_intercept=False).fit(X_selected, y_train)
    else:
        model = LinearRegression().fit(X_selected, y_train)

    #Calculating train set error
    pred_in = model.predict(X_selected)
    MAE_train = mean_absolute_error(y_train, pred_in)



    #Calculating CV set error
    tscv = TimeSeriesSplit(n_splits=5)
    MAE_CV_list = []
    RAE_CV_list = []
    RAE_in_list = []
    for train_index, test_index in tscv.split(y_train):
        X_CV_train, X_CV_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]

        model_CV = LinearRegression().fit(X_CV_train, y_CV_train)
        y_CV_pred = model_CV.predict(X_CV_test)
        MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
        MAE_CV_list.append(MAE_CV)
        RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
        RAE_CV_list.append(RAE_CV)
    
    MAE_CV = statistics.mean(MAE_CV_list)


    




    #Calculating the error on the test set
    X_selected_test = X_test[selected_names]
    pred_out = model.predict(X_selected_test)
    MAE_test = mean_absolute_error(y_test, pred_out)






    #Deleting the lags of the not selected variables
    #List of selected variables without the Lag notation at the end
    selected_names_without_L = []
    for name in selected_names:
        name_without_L = name.split(".L")[0]
        selected_names_without_L.append(name_without_L)




    Xs = [X_train_all_lags, X_test_all_lags]
    split = ["train", "test"]
    Xs_all_lags = {}
    counter = 0
    for X in Xs:

        X_all_lags_names = list(X.columns) 
        total_n_names = len(X_all_lags_names)
        n_unique_vars = int(total_n_names/(max_lag+1))
        n_unique_vars = list(range(1, n_unique_vars+1))
        for n in n_unique_vars:
            start_index = n*(max_lag+1) - (max_lag+1)
            last_index = n*(max_lag+1)
            unique_var_lags = X_all_lags_names[start_index:last_index]
            unique_var_name = unique_var_lags[0].split(".L")[0]
            if unique_var_name not in selected_names_without_L:
                for lag in unique_var_lags:
                        X.pop(lag)
            else:
                continue

        Xs_all_lags[split[counter]] = X
        counter += 1
    
    X_train_selected_all_lags = Xs_all_lags["train"]
    X_test_selected_all_lags = Xs_all_lags["test"]


    #Truncating y_train so that its length corresponds to that of Xs
    y_train=y_train.iloc[max_lag:]
    y_train.reset_index(drop=True, inplace=True)








    #Fitting the model on all features
    model = AutoReg(y_train, lags=0, exog=X_train_selected_all_lags).fit()

   
    # Using backward elimination to drop insignificant features
    # Defining critiacl p-value determining whether a feture is to be dropped
    critical_p_value = 0.05
    # Finding p-value of the lesat siginificant feature
    max_p_value = max(model.pvalues)
    # Defining const_dropped to know whether we run Autoreg with or w/o const
    const_dropped = False
    while max_p_value >= critical_p_value:
        # Column name of the least significant feature
        least_sig_var = list(model.params[np.where(model.pvalues == max_p_value)[0]].index)[0]
        # If least_sig_var is the constant we run Autoreg without it
        if least_sig_var == "const":
            model = AutoReg(y_train, lags=0, exog=X_train_selected_all_lags, trend="n").fit()
            const_dropped = True

        else:
            # Dropping the least_sig_var from the df
            X_train_selected_all_lags.pop(least_sig_var)
            # If const has been dropped, we run Autoreg w/o it
            if const_dropped:
                try:
                    model = AutoReg(y_train, lags=0, exog=X_train_selected_all_lags, trend="n").fit()
                except ValueError:
                    print("\n\n\n\nNo coefficients appear to be significant in the estimated model.\n\n\n\n")
            else:
                model = AutoReg(y_train, lags=0, exog=X_train_selected_all_lags).fit()

        # At the end of each iteration we find the new highest p-value        
        max_p_value = max(model.pvalues)

    # Defining list of all significant variables (except for const - because it is not in the data)
    selected_names_new = [n for n in list(model.params.index) if n!= "const"]
    X_selected = X_train_selected_all_lags
    if const_dropped:
        model = LinearRegression(fit_intercept=False).fit(X_selected, y_train)
    else:
        model = LinearRegression().fit(X_selected, y_train)

    




    #Calculating the train set error
    pred_in = model.predict(X_selected)
    MAE_train_all = mean_absolute_error(y_train, pred_in)





    #Calculating the CV set error
    tscv = TimeSeriesSplit(n_splits=5)
    MAE_CV_list = []
    RAE_CV_list = []
    RAE_in_list = []
    for train_index, test_index in tscv.split(y_train):
        X_CV_train, X_CV_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]

        model_CV = LinearRegression().fit(X_CV_train, y_CV_train)
        y_CV_pred = model_CV.predict(X_CV_test)
        MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
        MAE_CV_list.append(MAE_CV)
        RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
        RAE_CV_list.append(RAE_CV)
    
    MAE_CV_all = statistics.mean(MAE_CV_list)









    #Deleting the not selected lags
    X_test_selected_all_lags = X_test_selected_all_lags[selected_names_new]
    #Truncating the length of y so that it corresponds to that of X
    y_test = y_test.iloc[max_lag:]
    y_test.reset_index(drop=True, inplace=True)




    #Calculating the train set error
    pred_out = model.predict(X_test_selected_all_lags)
    MAE_test_all = mean_absolute_error(y_test, pred_out)

    
    
    # Recording end time
    end = time.time()
    time_to_run = end - start

    # Saving the selected subsets (evoML requires special formatiing, so it is saved seperately)
    selected_df_evoML = lag_n_times_evoML(df, max_lag+1, target=target, selected_f_names=selected_names_new)
    selected_df = lag_n_times(df, max_lag+1, target=target, selected_f_names=selected_names_new)



    return selected_names, selected_names_new, [MAE_train, MAE_CV, MAE_test], [MAE_train_all, MAE_CV_all, MAE_test_all], selected_df_evoML, selected_df, time_to_run








def FC(df, target, test_size, n_features, n_features_2, max_lag):

    # Recording start time
    start = time.time()

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

    # for feature in features:
    #     result = adfuller(staionarity_df[feature], autolag="t-stat", regression="c")
    #     counter = 0

    #     while result[1] >= 0.01:
    #         staionarity_df[feature] = staionarity_df[feature] - staionarity_df[feature].shift(1)
    #         counter += 1
    #         #dropna(inplace=False) because it drops one observation for each feature
    #         result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat", regression="c")
    #     print(f'Order of integration for feature "{feature}" is {counter}')
    #     orders_of_integ[feature] = counter

    # staionarity_df.dropna(inplace=True)
    # staionarity_df.reset_index(drop=True, inplace=True)
    # y_train = staionarity_df[target]
    # X_train = staionarity_df[[n for n in list(staionarity_df.columns) if n != target]]








    # # Stationarising test data
    # stationarity_df_test = pd.concat([y_test, X_test], axis=1)
    # features = list(stationarity_df_test.columns)
    # for feature in features:
    #     # Continue if the feature was found to be stationary without tranformation
    #     if orders_of_integ[feature] == 0:
    #         continue
    #     else:
    #         order = orders_of_integ[feature]
    #         integr_list = list(range(order, order+1))
    #         # Difference o times as with the train data
    #         for o in integr_list:
    #             stationarity_df_test[feature] = stationarity_df_test[feature].diff()
    # stationarity_df_test.dropna(inplace=True)
    # stationarity_df_test.reset_index(drop=True, inplace=True)
    # y_test = stationarity_df_test[target]
    # X_test = stationarity_df_test[[n for n in list(stationarity_df_test.columns) if n != target]]

   

    # Creating datasets with additional 19 lags for all features (to be used later)
    Xs = [X_train, X_test]
    split = ["train", "test"]
    Xs_all_lags = {}
    counter = 0
    for X in Xs:
        var_names = list(X.columns)

        names_without_L = []
        X_all_lags = []
        for name in var_names:
            name_without_L = name.split(".L")[0]
            names_without_L.append(name_without_L)

            columns = []
            for i in list(range(2, max_lag+1+1)):
                columns.append(name_without_L + ".L"+str(i))

            feature_n_df = pd.DataFrame(columns=columns)
            for i in list(range(max_lag)):
                feature_n_df[columns[i]] = X[name].shift(i+1)

            feature_n_df = pd.concat([X[name], feature_n_df], axis=1)
            X_all_lags.append(feature_n_df)

        X_all_lags = pd.concat(X_all_lags, axis=1)
        X_all_lags = X_all_lags.iloc[max_lag:]
        X_all_lags.reset_index(drop=True, inplace=True)
        Xs_all_lags[split[counter]] = X_all_lags
        counter += 1

    X_train_all_lags = Xs_all_lags["train"]
    X_test_all_lags = Xs_all_lags["test"]





    #Performing first stage feature selection
    X_train_names = list(X_train.columns)
    X_train_np = X_train.to_numpy()
    # y_train_names = list(y_train.columns)
    y_train_np = y_train.to_numpy()
    ranks = fisher_score.fisher_score(X_train_np, y_train_np)
    idx_selected = fisher_score.feature_ranking(ranks)
    selected_names = list(X_train_names[n] for n in idx_selected[0:n_features])
    X_selected = X_train[selected_names]






    #fitting the model on the unique selected features
    model = LinearRegression().fit(X_selected, y_train)









    #Calculating train set error
    pred_in = model.predict(X_selected)
    MAE_train = mean_absolute_error(y_train, pred_in)








    #Calculating CV set error
    tscv = TimeSeriesSplit(n_splits=5)
    MAE_CV_list = []
    RAE_CV_list = []
    RAE_in_list = []
    for train_index, test_index in tscv.split(y_train):
        X_CV_train, X_CV_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]

        model_CV = LinearRegression().fit(X_CV_train, y_CV_train)
        y_CV_pred = model_CV.predict(X_CV_test)
        MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
        MAE_CV_list.append(MAE_CV)
        RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
        RAE_CV_list.append(RAE_CV)
    
    MAE_CV = statistics.mean(MAE_CV_list)


    




    #Calculating the error on the test set
    X_selected_test = X_test[selected_names]
    pred_out = model.predict(X_selected_test)
    MAE_test = mean_absolute_error(y_test, pred_out)






    #Deleting the lags of the not selected variables
    #List of selected variables without the Lag notation at the end
    selected_names_without_L = []
    for name in selected_names:
        name_without_L = name.split(".L")[0]
        selected_names_without_L.append(name_without_L)




    Xs = [X_train_all_lags, X_test_all_lags]
    split = ["train", "test"]
    Xs_all_lags = {}
    counter = 0
    for X in Xs:

        X_all_lags_names = list(X.columns) 
        total_n_names = len(X_all_lags_names)
        n_unique_vars = int(total_n_names/(max_lag+1))
        n_unique_vars = list(range(1, n_unique_vars+1))
        for n in n_unique_vars:
            start_index = n*(max_lag+1) - (max_lag+1)
            last_index = n*(max_lag+1)
            unique_var_lags = X_all_lags_names[start_index:last_index]
            unique_var_name = unique_var_lags[0].split(".L")[0]
            if unique_var_name not in selected_names_without_L:
                for lag in unique_var_lags:
                        X.pop(lag)
            else:
                continue

        Xs_all_lags[split[counter]] = X
        counter += 1
    
    X_train_selected_all_lags = Xs_all_lags["train"]
    X_test_selected_all_lags = Xs_all_lags["test"]


    #Truncating y_train so that its length corresponds to that of Xs
    y_train=y_train.iloc[max_lag:]
    y_train.reset_index(drop=True, inplace=True)







    #Performing feature seleection on the datasets with all the lags of the selected features
    X_train_selected_all_lags_names = list(X_train_selected_all_lags.columns)
    X_train_selected_all_lags_np = X_train_selected_all_lags.to_numpy()
    y_train_np = y_train.to_numpy()
    ranks = fisher_score.fisher_score(X_train_selected_all_lags_np, y_train_np)
    idx_selected = fisher_score.feature_ranking(ranks)
    selected_names_new = list(X_train_selected_all_lags_names[n] for n in idx_selected[0:n_features_2])
    X_selected = X_train_selected_all_lags[selected_names_new]






    #Fitting the model on the selected lags
    model = LinearRegression().fit(X_selected, y_train)


    




    #Calculating the train set error
    pred_in = model.predict(X_selected)
    MAE_train_all = mean_absolute_error(y_train, pred_in)





    #Calculating the CV set error
    tscv = TimeSeriesSplit(n_splits=5)
    MAE_CV_list = []
    RAE_CV_list = []
    RAE_in_list = []
    for train_index, test_index in tscv.split(y_train):
        X_CV_train, X_CV_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]

        model_CV = LinearRegression().fit(X_CV_train, y_CV_train)
        y_CV_pred = model_CV.predict(X_CV_test)
        MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
        MAE_CV_list.append(MAE_CV)
        RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
        RAE_CV_list.append(RAE_CV)
    
    MAE_CV_all = statistics.mean(MAE_CV_list)









    #Deleting the not selected lags
    X_test_selected_all_lags = X_test_selected_all_lags[selected_names_new]
    #Truncating the length of y so that it corresponds to that of X
    y_test = y_test.iloc[max_lag:]
    y_test.reset_index(drop=True, inplace=True)




    #Calculating the train set error
    pred_out = model.predict(X_test_selected_all_lags)
    MAE_test_all = mean_absolute_error(y_test, pred_out)


    # Recording end time
    end = time.time()
    time_to_run = end - start


    # Saving the selected subsets (evoML requires special formatiing, so it is saved seperately)
    selected_df_evoML = lag_n_times_evoML(df, max_lag+1, target=target, selected_f_names=selected_names_new)
    selected_df = lag_n_times(df, max_lag+1, target=target, selected_f_names=selected_names_new)



    return selected_names, selected_names_new, [MAE_train, MAE_CV, MAE_test], [MAE_train_all, MAE_CV_all, MAE_test_all], selected_df_evoML, selected_df, time_to_run
