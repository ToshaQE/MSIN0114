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
# import lightgbm as lgb
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
# logging.INFO

import pmdarima as pmd

from my_metrics import rae, rrse
from lag_functions import lag_n_times
import time

#DISREGARD STATIONARITY METHOD - SHOULD ALWAYS BE SET TO 0
#DF - Dataset with target amd all fetures
#target - target feature name
#stationarity method - disregard
#test size - proprtion of the data saved for testing



def stage_one(df, target, max_lag, stationarity_method, test_size):

    start = time.time()


    # Step 1: Tranformation for stationarity d
    # Here features are everything except for the date
    feature_df = df.loc[:, ~df.columns.isin([target, "Date"])]
    target_df = df.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=test_size, shuffle=False)
    
    # Resetting index for later modelling purposes
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Dataframe to perform ADF test
    staionarity_df = pd.concat([y_train, X_train], axis=1)

    features = list(staionarity_df.columns)

    # features = [n for n in list(X_train.columns) if n != "Date"]
    
    # Coppying dataframes for stationarity and back-transformation purposes
    yx_train_original = staionarity_df.copy()
    y_original = y_train.copy()
    y_logged = np.log(y_original)

    orders_of_integ = {}
    const_counters = {}

    for feature in features:
        result = adfuller(staionarity_df[feature], autolag="t-stat", regression="c")
        counter = 0
        if stationarity_method == 0:
            while result[1] >= 0.01:
                staionarity_df[feature] = staionarity_df[feature] - staionarity_df[feature].shift(1)
                #df_small.dropna()
                counter += 1
                #dropna(inplace=False) because it drops one observation for each feature
                result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat", regression="c")
            print(f'Order of integration for feature "{feature}" is {counter}')
            orders_of_integ[feature] = counter
        elif stationarity_method == 1:
            feature_logged = np.log(staionarity_df[feature])
            inf_count = np.isinf(feature_logged).sum()
            const_counter = 0
            # If inf count is greater than 0 it is likely that original series contains 0s or negative values
            # Hence we add a constant to the series and only then apply log tranformations until there are no zeroes/nrgative values
            while inf_count > 0:
                if const_counter == 0:
                    feature_with_constant = staionarity_df[feature] + 1
                    feature_logged = np.log(feature_with_constant)
                    inf_count = np.isinf(feature_logged).sum()
                    const_counter += 1
                else:
                    feature_with_constant = feature_with_constant + 1
                    feature_logged = np.log(feature_with_constant)
                    inf_count = np.isinf(feature_logged).sum()
                    const_counter += 1

            while result[1] >= 0.01:
                feature_differenced = feature_logged.diff()
                staionarity_df[feature] = feature_differenced
                #df_small.dropna()
                counter += 1
                #dropna(inplace=False) because it drops one observation for each feature
                result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat")
            print(f'Order of integration for feature "{feature}" is {counter}')
            orders_of_integ[feature] = counter
            const_counters[feature] = const_counter

    # staionarity_df[target] = y_original

    staionarity_df.dropna(inplace=True)
    staionarity_df.reset_index(drop=True, inplace=True)

    y_train = staionarity_df[target]
    X_train = staionarity_df[[n for n in list(staionarity_df.columns) if n != target]]

    #Error Correction Model
    cointegr_dict = {}
    ECM_residuals = {}
    ECM_OLSs = {}
    x_features = list(X_train.columns)

    for x_feature in x_features:
        if orders_of_integ[target] == orders_of_integ[x_feature] and orders_of_integ[target] != 0:
            E_G = engle_granger(yx_train_original[target], yx_train_original[x_feature], trend = "n", method = "t-stat")
            if E_G.pvalue < 0.05:
                cointegr_dict[x_feature] = 1
                ECM_X_const = sm.add_constant(yx_train_original[x_feature])
                ECM_1 = OLS(yx_train_original[target], ECM_X_const, hasconst=True).fit(cov_type=("HC0"))
                ECM_res = ECM_1.resid
                ECM_res = ECM_res.shift(1).dropna().reset_index(drop=True)
                ECM_res = ECM_res.iloc[max_lag:].reset_index(drop=True)
                ECM_residuals[x_feature] = ECM_res
                ECM_OLSs[x_feature] = ECM_1

            # ECM_1 = OLS(yx_train_original[target], yx_train_original[feature]).fit(cov_type=("HC0"))
            # ECM_res = ECM_1.resid
            # ECM_adf = adfuller(ECM_res, autolag = "t-stat", regrression = "nc")[1]

            







    # Step 2: Building a univariate model and finding the optimal l

    BICs = []
    for i in list(range(max_lag)):
        model = AutoReg(y_train, lags=i).fit()
        BICs.append(model.hqic)

    min_bic_ind = BICs.index(min(BICs))

    # model = AutoReg(df_small.iloc[:,1], lags=min_bic_ind).fit()
    # model.summary()

    # Due to statsmodels weird properties, you can not test trained model on unseen y-data, but only on unseen X-data.
    # Hence we need to perform some data manipulations to make the testing possible.

    columns_y = []
    for i in list(range(1, min_bic_ind+1)):
        columns_y.append(target+".L"+str(i))

    y_lags_df = pd.DataFrame(columns=columns_y)
    for i in list(range(min_bic_ind)):
        y_lags_df[columns_y[i]] = y_train.shift(i+1)

    # Truncating lags of y at the maximum lag length
    # y_lags_df.fillna(1, inplace=True)
    y_lags_df = y_lags_df.iloc[max_lag:,:]
    y_lags_df.reset_index(drop=True, inplace=True)

    # Step 2: Bulding augmented model and finding the optimal w for each Xi
    
    Xs = list(X_train.columns)

    # Truncating y_train for model training at max_lag length
    y_train_m = y_train.iloc[max_lag:]
    y_train_m.reset_index(drop=True, inplace=True)

    # Defining dictionary to store all augmented models
    aug_models = {}
    feature_n_dfs = {}
    feature_n_dfs_merge = [y_lags_df]
    n_lags_for_xi = {}
    
    for x in Xs:
        columns = []
        for i in list(range(1, max_lag+1)):
            columns.append(x + ".L"+str(i))

        feature_n_df = pd.DataFrame(columns=columns)
        for i in list(range(max_lag)):
            feature_n_df[columns[i]] = X_train[x].shift(i+1)

        feature_n_df = feature_n_df.iloc[max_lag:,:]
        feature_n_df.reset_index(drop=True, inplace=True)
        y_and_x_lags_df = pd.concat([y_lags_df, feature_n_df], axis=1)

        BICs = []
        #Why do I have max_lag-1 and then i+1?
        # +1 is to not make X lags = 0
        # y_and_x_lags_df_m = y_and_x_lags_df.iloc[:,:i+len(list(y_lags_df.columns))+1]
        #y_and_x_lags_df.reset_index(drop=True, inplace=True)
        for i in list(range(max_lag-1)):
            model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df.iloc[:,:i+len(list(y_lags_df.columns))+1]).fit()
            BICs.append(model.bic)

        min_bic_ind_aug = BICs.index(min(BICs))
        feature_n_df1 = y_and_x_lags_df
        y_and_x_lags_df = y_and_x_lags_df.iloc[:,:min_bic_ind_aug+len(list(y_lags_df.columns))+1]
        y_and_x_lags_df.reset_index(drop=True, inplace=True)




        #Sequential t-testing to keep only significant lags
        model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()
        not_sig_lags = []
        # Defining critiacl p-value determining whether a feture is to be dropped
        critical_p_value = 0.05
        # Finding p-value of the least siginificant non-target feature
        regressors = list(model.pvalues.index)
        non_target_regressors = []
        for r in regressors:
            original_feature_name = r.split(".")[0]
            if original_feature_name != target and original_feature_name != "const":
                non_target_regressors.append(r)
            else:
                continue
        
        non_target_exists = len(non_target_regressors) != 0
        non_target_pvalues = model.pvalues.loc[non_target_regressors]
        max_p_value = max(non_target_pvalues)
        least_sig_var = list(non_target_pvalues[non_target_pvalues == max_p_value].index)[0]

        #while the max p value is greater then the critical value
        #we drop the respective non-target variable and re-estimate the model
        #Backward t-testing
        while max_p_value >= critical_p_value:
            y_and_x_lags_df.pop(least_sig_var)
            model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()

            # At the end of each iteration we find the new highest p-value
            regressors = list(model.pvalues.index)
            non_target_regressors = []
            for r in regressors:
                original_feature_name = r.split(".")[0]
                if original_feature_name != target and original_feature_name != "const":
                    non_target_regressors.append(r)
                else:
                    continue
            
            non_target_exists = len(non_target_regressors) != 0

            #Check if any non-target still present in the regression             
            if non_target_exists == False:
                break
            else:
                non_target_pvalues = model.pvalues.loc[non_target_regressors]
                max_p_value = max(non_target_pvalues)
                least_sig_var = list(non_target_pvalues[non_target_pvalues == max_p_value].index)[0]

        # If no lags of x have significant coefficients then x does not granger cause Y 
        if non_target_exists == False:
            continue
        
        else:

            # Adding ECM residual to the feature n dataset if it appears to be cointegrated with the target
            if x in list(ECM_residuals.keys()):
                ECM_res_name = x + "_ECM_Res.L1"
                y_and_x_lags_df[ECM_res_name] = ECM_residuals[x]

            model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()




            #Testing the cause-effect relationship
            gr_test_df = pd.concat([X_train[x], y_train], axis=1)
            granger_p_stat = grangercausalitytests(gr_test_df, maxlag=[min_bic_ind+1])[min_bic_ind+1][0]['params_ftest'][1]
            if granger_p_stat >= 0.05:
                aug_models[x] = model
                feature_n_dfs[x] = feature_n_df1
                feature_n_dfs_merge.append(y_and_x_lags_df.iloc[:,len(list(y_lags_df.columns)):])
                n_lags_for_xi[x] = min_bic_ind_aug + 1
                #model.summary()
            elif granger_p_stat >= 0.01:
                print(f'\n\nGranger causality from "{target}" to "{x}" can not be rejected with a p-value={granger_p_stat:.3}')
            else:
                continue

    
    try:
        
        feature_n_dfs_merge = pd.concat(feature_n_dfs_merge, axis=1)

        if len(feature_n_dfs_merge) == 0:
                    print("\n\n\n\nZero lags of y have been selected and H0 of reverse causlity could not be rejected for any X.\n\n\n\n")

        fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge).fit()



        # Defining list of all significant variables (except for const - because it is not in the data)
        names_of_sig_vars = [n for n in list(fin_model.params.index) if n!= "const"]


        y_pred_in = fin_model.predict()
        MAE_train = np.nanmean(abs(y_pred_in - y_train_m))
        MSE_train = mean_squared_error(y_pred_in, y_train_m)
        RMSE_train = math.sqrt(MSE_train)
        RAE_train = rae(actual=y_train_m, predicted = y_pred_in)
        RRSE_train = rrse(actual=y_train_m, predicted = y_pred_in)

        if orders_of_integ[target] > 0:

            # Calculating train scores in the original scale

            if stationarity_method == 0:
                y_train_m_destat = y_train_m.copy()
                y_train_m_destat.loc[-1] = y_original.iloc[max_lag]
                y_train_m_destat.index = y_train_m_destat.index + 1
                y_train_m_destat = y_train_m_destat.sort_index()
                y_train_m_destat = y_train_m_destat.cumsum()


                y_pred_in_destat = y_pred_in.copy()
                y_pred_in_destat.loc[-1] = y_original.iloc[max_lag]
                y_pred_in_destat.index = y_pred_in_destat.index + 1
                y_pred_in_destat = y_pred_in_destat.sort_index()
                y_pred_in_destat = y_pred_in_destat.cumsum()

                MAE_train_destat = np.nanmean(abs(y_pred_in_destat - y_train_m_destat))

            elif stationarity_method == 1:
                y_train_m_destat = y_train_m.copy()
                y_train_m_destat.loc[-1] = y_logged.iloc[max_lag]
                y_train_m_destat.index = y_train_m_destat.index + 1
                y_train_m_destat = y_train_m_destat.sort_index()
                y_train_m_destat = np.exp(y_train_m_destat.cumsum())


                y_pred_in_destat = y_pred_in.copy()
                y_pred_in_destat.loc[-1] = y_logged.iloc[max_lag]
                y_pred_in_destat.index = y_pred_in_destat.index + 1
                y_pred_in_destat = y_pred_in_destat.sort_index()
                y_pred_in_destat = np.exp(y_pred_in_destat.cumsum())

                MAE_train_destat = np.nanmean(abs(y_pred_in_destat - y_train_m_destat))




        tscv = TimeSeriesSplit(n_splits=10)
        MAE_CV_list = []
        RAE_CV_list = []
        RAE_in_list = []
        for train_index, test_index in tscv.split(y_train_m):
            X_CV_train, X_CV_test = feature_n_dfs_merge.iloc[train_index], feature_n_dfs_merge.iloc[test_index]
            y_CV_train, y_CV_test = y_train_m.iloc[train_index], y_train_m.iloc[test_index]

            model_CV = AutoReg(y_CV_train, lags=0, exog=X_CV_train).fit()
            y_CV_pred = model_CV.predict(start=test_index[0], end=test_index[-1], exog_oos = X_CV_test)
            MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
            MAE_CV_list.append(MAE_CV)
            RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
            RAE_CV_list.append(RAE_CV)

        MAE_CV = statistics.mean(MAE_CV_list)








        # Coppying data for ECM imlplementation
        y_test_non_stat = y_test.copy()
        X_test_non_stat = X_test.copy()

        # Stationarising test data
        stationarity_df_test = pd.concat([y_test, X_test], axis=1)
        features = list(stationarity_df_test.columns)

        if stationarity_method == 0:
        # if transformation is simple differencing
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

        elif stationarity_method == 1:
        # if transformation is log differencing
            for feature in features:
                # Continue if the feature was found to be stationary withoud tranformation
                if orders_of_integ[feature] == 0:
                        continue
                else:
                    # Check whether any constants were added to the training data
                    if const_counters[feature] > 0:
                        stationarity_df_test[feature] = stationarity_df_test[feature] + const_counters[feature]
                    # Logging the data
                    stationarity_df_test[feature] = np.log(stationarity_df_test[feature])
                    order = orders_of_integ[feature]
                    integr_list = list(range(order, order+1))
                    # Difference o times as with the train data
                    for o in integr_list:
                        stationarity_df_test[feature] = stationarity_df_test[feature].diff()

        # stationarity_df_test[target] = y_test_non_stat

        stationarity_df_test.dropna(inplace=True)
        stationarity_df_test.reset_index(drop=True, inplace=True)

        y_test = stationarity_df_test[target]
        X_test = stationarity_df_test[[n for n in list(stationarity_df_test.columns) if n != target]]

        # Formatting the test dataframes to suit the model's exog format
        test_data = []


        #Finding the maximum seleceted lag length to truncate the test data appropriately
        selected_lag_lens = []
        selected_lag_lens.append(min_bic_ind)
        for x_name, lag_len in n_lags_for_xi.items():
            selected_lag_lens.append(lag_len)

        max_sel_lag = max(selected_lag_lens)

        # Formatting y
        y_lags_df = pd.DataFrame(columns=columns_y)
        for i in list(range(min_bic_ind)):
            y_lags_df[columns_y[i]] = y_test.shift(i+1)

        # Truncating lags of y at the maximum lag length
        y_lags_df = y_lags_df.iloc[max_sel_lag:,:]
        y_lags_df.reset_index(drop=True, inplace=True)

        test_data.append(y_lags_df)


        #Formatting Xs and implementing ECM on the test data
        Cointegrated_Xs = list(ECM_residuals.keys())

        for x_name, lag_len in n_lags_for_xi.items():
            columns = []
            for i in list(range(1, lag_len+1)):
                columns.append(x_name+".L"+str(i))

            feature_x_df = pd.DataFrame(columns=columns)
            for i in list(range(lag_len)):
                feature_x_df[columns[i]] = X_test[x_name].shift(i+1)
            
            feature_x_df = feature_x_df.iloc[max_sel_lag:,:]
            feature_x_df.reset_index(drop=True, inplace=True)

            if x_name in Cointegrated_Xs:
                ECM_OLS = ECM_OLSs[x_name]
                X_test_ECM = sm.add_constant(X_test_non_stat[x_name])
                y_ECM_pred = ECM_OLS.predict(X_test_ECM)
                y_test_ECM_res = y_ECM_pred - y_test_non_stat
                y_test_ECM_res = y_test_ECM_res.shift(1)
                y_test_ECM_res = y_test_ECM_res.iloc[max_sel_lag:]
                y_test_ECM_res.reset_index(drop=True, inplace=True)
                feature_x_df[x_name + "_ECM_Res.L1"] = y_test_ECM_res
            
            test_data.append(feature_x_df)
    
        # Merging y and Xs
        test_data = pd.concat(test_data, axis=1)
        # Only keeping the significant features
        test_data = test_data[names_of_sig_vars]
        # Truncating y_test, so its length corresponds to that of y_train_m
        y_test = y_test.iloc[max_sel_lag:]
        y_test.reset_index(drop=True, inplace=True)

        first_oos_ind = len(y_train_m)
        last_oos_ind = first_oos_ind + len(y_test) - 1
        y_pred_out = fin_model.predict(start=first_oos_ind, end=last_oos_ind, exog_oos=test_data)
        y_pred_out.reset_index(drop=True, inplace=True)

        MAE_test = np.nanmean(abs(y_pred_out - y_test))
        MSE_test = mean_squared_error(y_pred_out, y_test)
        RMSE_test = math.sqrt(MSE_test)
        RAE_test = rae(actual=y_test, predicted = y_pred_out)
        RRSE_test = rrse(actual=y_test, predicted = y_pred_out)



        if orders_of_integ[target] > 0:

            # Calculating test scores in the original scale
            if stationarity_method == 0:
                y_pred_out_destat = y_pred_out.copy()
                y_test_non_stat_destat = y_test_non_stat.copy()
                y_test_non_stat_destat = y_test_non_stat_destat.iloc[max_sel_lag:]
                y_test_non_stat_destat.reset_index(drop=True, inplace=True)

                y_pred_out_destat = y_test_non_stat_destat.iloc[:-1] + y_pred_out

                y_test_non_stat_destat = y_test_non_stat_destat.iloc[1:]
                y_test_non_stat_destat.reset_index(drop=True, inplace=True)

                MAE_test_destat = np.nanmean(abs(y_pred_out_destat - y_test_non_stat_destat))
                RAE_test = rae(actual=y_test_non_stat_destat, predicted = y_pred_out_destat)

            elif stationarity_method == 1:
                y_test_logged = np.log(y_test_non_stat)
                y_pred_out_destat = y_pred_out.copy()
                y_pred_out_destat.loc[-1] = y_test_logged.iloc[max_sel_lag]
                y_pred_out_destat.index = y_pred_out_destat.index + 1
                y_pred_out_destat = y_pred_out_destat.sort_index()
                y_pred_out_destat = np.exp(y_pred_out_destat.cumsum())

                y_test_non_stat_destat = y_test_logged.copy()
                y_test_non_stat_destat = y_test_non_stat_destat.iloc[max_sel_lag:]
                y_test_non_stat_destat.reset_index(drop=True, inplace=True)
                y_test_non_stat_destat = np.exp(y_test_non_stat_destat)

                MAE_test_destat = np.nanmean(abs(y_pred_out_destat - y_test_non_stat_destat))

            MAE = {"train": MAE_train_destat, "test": MAE_test_destat}
        else:
            MAE = {"train": MAE_train, "test": MAE_test}
            y_train_m_destat = y_train_m
            y_test_non_stat_destat = y_test




        
        MAE_stat = {"train": MAE_train, "test": MAE_test}
        MAE_ = {"Original": MAE, "Stationary": MAE_stat}
        logging.info("Check")






        #Non-stationary dataset with selected features only, to be used in evoML
        selected_vars = list(feature_n_dfs_merge.columns)
        names_without_L = []
        for var in selected_vars:
            name_without_L = var.split(".")[0]
            if name_without_L != "const" and (name_without_L in names_without_L)==False:
                names_without_L.append(name_without_L)
            else:
                continue

        only_seleceted_vars_df = df[names_without_L]
        selected_dfs_lags = []
        for var in names_without_L:
            columns = []
            for i in list(range(1, max_lag+1)):
                columns.append(var + ".L"+str(i))

            selcted_var_lag_df = pd.DataFrame(columns=columns)
            for i in list(range(max_lag)):
                selcted_var_lag_df[columns[i]] = only_seleceted_vars_df[var].shift(i+1)
            
            selected_dfs_lags.append(selcted_var_lag_df)
        selected_dfs_lags = pd.concat(selected_dfs_lags, axis=1)
        selected_features_df = selected_dfs_lags[selected_vars]
        
        export_df = pd.concat([df[["Date", target]], selected_features_df], axis=1)
        export_df = export_df.iloc[max_lag:,:]
        export_df.reset_index(inplace=True, drop=True)

        export_df_evoml = pd.concat([df[["Date", target]], selected_features_df], axis=1)
        export_df_evoml[target] = export_df_evoml[target].shift(1)
        export_df_evoml.pop(target+".L1")
        export_df_evoml["Date"] = export_df_evoml["Date"].shift(1)
        export_df_evoml = export_df_evoml.iloc[max_lag:,:]
        export_df_evoml.reset_index(inplace=True, drop=True)


        end = time.time()
        time_to_run = end - start


        #Saving the test and train metrics 
        destat_data = {"y_train": y_train_m_destat, "y_test": y_test_non_stat_destat,
                        "stationarity_method": stationarity_method,
                        "y_integ_order": orders_of_integ[target]}
        my_metrics_test = {"MAE":[MAE_test], "RMSE":[RMSE_test], "RAE":[RAE_test], "RRSE":[RRSE_test]}
        my_metrics_train = {"MAE":[MAE_train], "RMSE":[RMSE_train], "RAE":[RAE_train], "RRSE":[RRSE_train]}
        my_metrics_CV = {"MAE": [MAE_CV]}
        my_metrics = {"train":my_metrics_train, "CV":my_metrics_CV, "test":my_metrics_test}

        Model_Data = Sun_Model(fin_model, fin_model.summary(), aug_models, MAE_,
                                y_train_m, feature_n_dfs_merge,
                                y_test, test_data,
                                y_pred_out, destat_data,
                                my_metrics,
                                export_df,
                                export_df_evoml,
                                time_to_run,
                                selected_vars)




        return Model_Data
    except ValueError:
        logging.error("Can not reject that the target variable 'reverse causes' independent features.")

























#Disregard
#Forces data not to be stationary. For apple data auto_arima finds that non-stationary Close provides better results

def stage_one_ns(df, target, max_lag, stationarity_method, test_size):

    start = time.time()


    # Step 1: Tranformation for stationarity d
    # Here features are everything except for the date
    feature_df = df.loc[:, ~df.columns.isin([target, "Date"])]
    target_df = df.loc[:, target]

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
        if stationarity_method == 0:
            while result[1] >= 0.01:
                staionarity_df[feature] = staionarity_df[feature] - staionarity_df[feature].shift(1)
                counter += 1
                result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat", regression="c")
            print(f'Order of integration for feature "{feature}" is {counter}')
            orders_of_integ[feature] = counter
        elif stationarity_method == 1:
            feature_logged = np.log(staionarity_df[feature])
            inf_count = np.isinf(feature_logged).sum()
            const_counter = 0
            # If inf count is greater than 0 it is likely that original series contains 0s or negative values
            # Hence we add a constant to the series and only then apply log tranformations until there are no zeroes/nrgative values
            while inf_count > 0:
                if const_counter == 0:
                    feature_with_constant = staionarity_df[feature] + 1
                    feature_logged = np.log(feature_with_constant)
                    inf_count = np.isinf(feature_logged).sum()
                    const_counter += 1
                else:
                    feature_with_constant = feature_with_constant + 1
                    feature_logged = np.log(feature_with_constant)
                    inf_count = np.isinf(feature_logged).sum()
                    const_counter += 1

            while result[1] >= 0.01:
                feature_differenced = feature_logged.diff()
                staionarity_df[feature] = feature_differenced
                #df_small.dropna()
                counter += 1
                #dropna(inplace=False) because it drops one observation for each feature
                result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat")
            print(f'Order of integration for feature "{feature}" is {counter}')
            orders_of_integ[feature] = counter
            const_counters[feature] = const_counter

    staionarity_df[target] = y_original

    staionarity_df.dropna(inplace=True)
    staionarity_df.reset_index(drop=True, inplace=True)

    y_train = staionarity_df[target]
    X_train = staionarity_df[[n for n in list(staionarity_df.columns) if n != target]]

    #Error Correction Model
    cointegr_dict = {}
    ECM_residuals = {}
    ECM_OLSs = {}
    x_features = list(X_train.columns)

    for x_feature in x_features:
        if orders_of_integ[target] == orders_of_integ[x_feature] and orders_of_integ[target] != 0:
            E_G = engle_granger(yx_train_original[target], yx_train_original[x_feature], trend = "n", method = "t-stat")
            if E_G.pvalue < 0.05:
                cointegr_dict[x_feature] = 1
                ECM_X_const = sm.add_constant(yx_train_original[x_feature])
                ECM_1 = OLS(yx_train_original[target], ECM_X_const, hasconst=True).fit(cov_type=("HC0"))
                ECM_res = ECM_1.resid
                ECM_res = ECM_res.shift(1).dropna().reset_index(drop=True)
                ECM_res = ECM_res.iloc[max_lag:].reset_index(drop=True)
                ECM_residuals[x_feature] = ECM_res
                ECM_OLSs[x_feature] = ECM_1

            # ECM_1 = OLS(yx_train_original[target], yx_train_original[feature]).fit(cov_type=("HC0"))
            # ECM_res = ECM_1.resid
            # ECM_adf = adfuller(ECM_res, autolag = "t-stat", regrression = "nc")[1]

            







    # Step 2: Building a univariate model and finding the optimal l

    BICs = []
    for i in list(range(max_lag)):
        model = AutoReg(y_train, lags=i).fit()
        BICs.append(model.hqic)

    min_bic_ind = BICs.index(min(BICs))

    # model = AutoReg(df_small.iloc[:,1], lags=min_bic_ind).fit()
    # model.summary()

    # Due to statsmodels weird properties, you can not test trained model on unseen y-data, but only on unseen X-data.
    # Hence we need to perform some data manipulations to make the testing possible.

    columns_y = []
    for i in list(range(1, min_bic_ind+1)):
        columns_y.append(target+".L"+str(i))

    y_lags_df = pd.DataFrame(columns=columns_y)
    for i in list(range(min_bic_ind)):
        y_lags_df[columns_y[i]] = y_train.shift(i+1)

    # Truncating lags of y at the maximum lag length
    # y_lags_df.fillna(1, inplace=True)
    y_lags_df = y_lags_df.iloc[max_lag:,:]
    y_lags_df.reset_index(drop=True, inplace=True)

    # Step 2: Bulding augmented model and finding the optimal w for each Xi
    
    Xs = list(X_train.columns)

    # Truncating y_train for model training at max_lag length
    y_train_m = y_train.iloc[max_lag:]
    y_train_m.reset_index(drop=True, inplace=True)

    # Defining dictionary to store all augmented models
    aug_models = {}
    feature_n_dfs = {}
    feature_n_dfs_merge = [y_lags_df]
    n_lags_for_xi = {}
    
    for x in Xs:
        columns = []
        for i in list(range(1, max_lag+1)):
            columns.append(x + ".L"+str(i))

        feature_n_df = pd.DataFrame(columns=columns)
        for i in list(range(max_lag)):
            feature_n_df[columns[i]] = X_train[x].shift(i+1)

        feature_n_df = feature_n_df.iloc[max_lag:,:]
        feature_n_df.reset_index(drop=True, inplace=True)
        y_and_x_lags_df = pd.concat([y_lags_df, feature_n_df], axis=1)

        BICs = []
        #Why do I have max_lag-1 and then i+1?
        # +1 is to not make X lags = 0
        # y_and_x_lags_df_m = y_and_x_lags_df.iloc[:,:i+len(list(y_lags_df.columns))+1]
        #y_and_x_lags_df.reset_index(drop=True, inplace=True)
        for i in list(range(max_lag-1)):
            model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df.iloc[:,:i+len(list(y_lags_df.columns))+1]).fit()
            BICs.append(model.bic)

        min_bic_ind_aug = BICs.index(min(BICs))
        feature_n_df1 = y_and_x_lags_df
        y_and_x_lags_df = y_and_x_lags_df.iloc[:,:min_bic_ind_aug+len(list(y_lags_df.columns))+1]
        y_and_x_lags_df.reset_index(drop=True, inplace=True)




        #Sequential t-testing to keep only significant lags
        model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()
        not_sig_lags = []
        # Defining critiacl p-value determining whether a feture is to be dropped
        critical_p_value = 0.05
        # Finding p-value of the least siginificant non-target feature
        regressors = list(model.pvalues.index)
        non_target_regressors = []
        for r in regressors:
            original_feature_name = r.split(".")[0]
            if original_feature_name != target and original_feature_name != "const":
                non_target_regressors.append(r)
            else:
                continue
        
        non_target_exists = len(non_target_regressors) != 0
        non_target_pvalues = model.pvalues.loc[non_target_regressors]
        max_p_value = max(non_target_pvalues)
        least_sig_var = list(non_target_pvalues[non_target_pvalues == max_p_value].index)[0]

        #while the max p value is greater then the critical value
        #we drop the respective non-target variable and re-estimate the model
        #Backward t-testing
        while max_p_value >= critical_p_value:
            y_and_x_lags_df.pop(least_sig_var)
            model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()

            # At the end of each iteration we find the new highest p-value
            regressors = list(model.pvalues.index)
            non_target_regressors = []
            for r in regressors:
                original_feature_name = r.split(".")[0]
                if original_feature_name != target and original_feature_name != "const":
                    non_target_regressors.append(r)
                else:
                    continue
            
            non_target_exists = len(non_target_regressors) != 0

            #Check if any non-target still present in the regression             
            if non_target_exists == False:
                break
            else:
                non_target_pvalues = model.pvalues.loc[non_target_regressors]
                max_p_value = max(non_target_pvalues)
                least_sig_var = list(non_target_pvalues[non_target_pvalues == max_p_value].index)[0]

        # If no lags of x have significant coefficients then x does not granger cause Y 
        if non_target_exists == False:
            continue
        
        else:

            # Adding ECM residual to the feature n dataset if it appears to be cointegrated with the target
            if x in list(ECM_residuals.keys()):
                ECM_res_name = x + "_ECM_Res.L1"
                y_and_x_lags_df[ECM_res_name] = ECM_residuals[x]

            model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()




            #Testing the cause-effect relationship
            gr_test_df = pd.concat([X_train[x], y_train], axis=1)
            granger_p_stat = grangercausalitytests(gr_test_df, maxlag=[min_bic_ind+1])[min_bic_ind+1][0]['params_ftest'][1]
            if granger_p_stat >= 0.05:
                aug_models[x] = model
                feature_n_dfs[x] = feature_n_df1
                feature_n_dfs_merge.append(y_and_x_lags_df.iloc[:,len(list(y_lags_df.columns)):])
                n_lags_for_xi[x] = min_bic_ind_aug + 1
                #model.summary()
            elif granger_p_stat >= 0.01:
                print(f'\n\nGranger causality from "{target}" to "{x}" can not be rejected with a p-value={granger_p_stat:.3}')
            else:
                continue


            # aug_models[features[n]] = model
            # feature_n_dfs[features[n]] = feature_n_df1
            # feature_n_dfs_merge.append(feature_n_df)
            # #model.summary()
    
    try:
        
        feature_n_dfs_merge = pd.concat(feature_n_dfs_merge, axis=1)

        if len(feature_n_dfs_merge) == 0:
                    print("\n\n\n\nZero lags of y have been selected and H0 of reverse causlity could not be rejected for any X.\n\n\n\n")

        fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge).fit()




        # # Using backward elimination to drop insignificant features
        # # Defining critiacl p-value determining whether a feture is to be dropped
        # critical_p_value = 0.05
        # # Finding p-value of the lesat siginificant feature
        # max_p_value = max(fin_model.pvalues)
        # # Defining const_dropped to know whether we run Autoreg with or w/o const
        # const_dropped = False
        # while max_p_value >= critical_p_value:
        #     # Column name of the least significant feature
        #     least_sig_var = list(fin_model.params[np.where(fin_model.pvalues == max_p_value)[0]].index)[0]
        #     # If least_sig_var is the constant we run Autoreg without it
        #     if least_sig_var == "const":
        #         fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge, trend="n").fit()
        #         const_dropped = True

        #     else:
        #         # Dropping the least_sig_var from the df
        #         feature_n_dfs_merge.pop(least_sig_var)
        #         # If const has been dropped, we run Autoreg w/o it
        #         if const_dropped:
        #             try:
        #                 fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge, trend="n").fit()
        #             except ValueError:
        #                 print("\n\n\n\nNo coefficients appear to be significant in the estimated model.\n\n\n\n")
        #         else:
        #             fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge).fit()

        #     # At the end of each iteration we find the new highest p-value        
        #     max_p_value = max(fin_model.pvalues)

        # Defining list of all significant variables (except for const - because it is not in the data)
        names_of_sig_vars = [n for n in list(fin_model.params.index) if n!= "const"]


        y_pred_in = fin_model.predict()
        MAE_train = np.nanmean(abs(y_pred_in - y_train_m))
        MSE_train = mean_squared_error(y_pred_in, y_train_m)
        RMSE_train = math.sqrt(MSE_train)
        RAE_train = rae(actual=y_train_m, predicted = y_pred_in)
        RRSE_train = rrse(actual=y_train_m, predicted = y_pred_in)

        if orders_of_integ[target] > 0:

            # Calculating train scores in the original scale

            if stationarity_method == 0:
                y_train_m_destat = y_train_m.copy()
                y_train_m_destat.loc[-1] = y_original.iloc[max_lag]
                y_train_m_destat.index = y_train_m_destat.index + 1
                y_train_m_destat = y_train_m_destat.sort_index()
                y_train_m_destat = y_train_m_destat.cumsum()


                y_pred_in_destat = y_pred_in.copy()
                y_pred_in_destat.loc[-1] = y_original.iloc[max_lag]
                y_pred_in_destat.index = y_pred_in_destat.index + 1
                y_pred_in_destat = y_pred_in_destat.sort_index()
                y_pred_in_destat = y_pred_in_destat.cumsum()

                MAE_train_destat = np.nanmean(abs(y_pred_in_destat - y_train_m_destat))

            elif stationarity_method == 1:
                y_train_m_destat = y_train_m.copy()
                y_train_m_destat.loc[-1] = y_logged.iloc[max_lag]
                y_train_m_destat.index = y_train_m_destat.index + 1
                y_train_m_destat = y_train_m_destat.sort_index()
                y_train_m_destat = np.exp(y_train_m_destat.cumsum())


                y_pred_in_destat = y_pred_in.copy()
                y_pred_in_destat.loc[-1] = y_logged.iloc[max_lag]
                y_pred_in_destat.index = y_pred_in_destat.index + 1
                y_pred_in_destat = y_pred_in_destat.sort_index()
                y_pred_in_destat = np.exp(y_pred_in_destat.cumsum())

                MAE_train_destat = np.nanmean(abs(y_pred_in_destat - y_train_m_destat))




        tscv = TimeSeriesSplit(n_splits=10)
        MAE_CV_list = []
        RAE_CV_list = []
        RAE_in_list = []
        for train_index, test_index in tscv.split(y_train_m):
            X_CV_train, X_CV_test = feature_n_dfs_merge.iloc[train_index], feature_n_dfs_merge.iloc[test_index]
            y_CV_train, y_CV_test = y_train_m.iloc[train_index], y_train_m.iloc[test_index]

            model_CV = AutoReg(y_CV_train, lags=0, exog=X_CV_train).fit()
            y_CV_pred = model_CV.predict(start=test_index[0], end=test_index[-1], exog_oos = X_CV_test)
            MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
            MAE_CV_list.append(MAE_CV)
            RAE_CV = rae(actual=y_CV_test, predicted = y_CV_pred)
            RAE_CV_list.append(RAE_CV)

        MAE_CV = statistics.mean(MAE_CV_list)








        # Coppying data for ECM imlplementation
        y_test_non_stat = y_test.copy()
        X_test_non_stat = X_test.copy()

        # Stationarising test data
        stationarity_df_test = pd.concat([y_test, X_test], axis=1)
        features = list(stationarity_df_test.columns)

        if stationarity_method == 0:
        # if transformation is simple differencing
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

        elif stationarity_method == 1:
        # if transformation is log differencing
            for feature in features:
                # Continue if the feature was found to be stationary withoud tranformation
                if orders_of_integ[feature] == 0:
                        continue
                else:
                    # Check whether any constants were added to the training data
                    if const_counters[feature] > 0:
                        stationarity_df_test[feature] = stationarity_df_test[feature] + const_counters[feature]
                    # Logging the data
                    stationarity_df_test[feature] = np.log(stationarity_df_test[feature])
                    order = orders_of_integ[feature]
                    integr_list = list(range(order, order+1))
                    # Difference o times as with the train data
                    for o in integr_list:
                        stationarity_df_test[feature] = stationarity_df_test[feature].diff()

        stationarity_df_test[target] = y_test_non_stat

        stationarity_df_test.dropna(inplace=True)
        stationarity_df_test.reset_index(drop=True, inplace=True)

        y_test = stationarity_df_test[target]
        X_test = stationarity_df_test[[n for n in list(stationarity_df_test.columns) if n != target]]

        # Formatting the test dataframes to suit the model's exog format
        test_data = []


        #Finding the maximum seleceted lag length to truncate the test data appropriately
        selected_lag_lens = []
        selected_lag_lens.append(min_bic_ind)
        for x_name, lag_len in n_lags_for_xi.items():
            selected_lag_lens.append(lag_len)

        max_sel_lag = max(selected_lag_lens)

        # Formatting y
        y_lags_df = pd.DataFrame(columns=columns_y)
        for i in list(range(min_bic_ind)):
            y_lags_df[columns_y[i]] = y_test.shift(i+1)

        # Truncating lags of y at the maximum lag length
        # y_lags_df.fillna(1, inplace=True)
        y_lags_df = y_lags_df.iloc[max_sel_lag:,:]
        y_lags_df.reset_index(drop=True, inplace=True)

        test_data.append(y_lags_df)


        #Formatting Xs and implementing ECM on the test data
        Cointegrated_Xs = list(ECM_residuals.keys())

        for x_name, lag_len in n_lags_for_xi.items():
            columns = []
            for i in list(range(1, lag_len+1)):
                columns.append(x_name+".L"+str(i))

            feature_x_df = pd.DataFrame(columns=columns)
            for i in list(range(lag_len)):
                feature_x_df[columns[i]] = X_test[x_name].shift(i+1)
            
            feature_x_df = feature_x_df.iloc[max_sel_lag:,:]
            feature_x_df.reset_index(drop=True, inplace=True)

            if x_name in Cointegrated_Xs:
                ECM_OLS = ECM_OLSs[x_name]
                X_test_ECM = sm.add_constant(X_test_non_stat[x_name])
                y_ECM_pred = ECM_OLS.predict(X_test_ECM)
                y_test_ECM_res = y_ECM_pred - y_test_non_stat
                y_test_ECM_res = y_test_ECM_res.shift(1)
                y_test_ECM_res = y_test_ECM_res.iloc[max_sel_lag:]
                y_test_ECM_res.reset_index(drop=True, inplace=True)
                feature_x_df[x_name + "_ECM_Res.L1"] = y_test_ECM_res
            
            test_data.append(feature_x_df)
    
        # Merging y and Xs
        test_data = pd.concat(test_data, axis=1)
        # Only keeping the significant features
        test_data = test_data[names_of_sig_vars]
        # Truncating y_test, so its length corresponds to that of y_train_m
        y_test = y_test.iloc[max_sel_lag:]
        y_test.reset_index(drop=True, inplace=True)

        first_oos_ind = len(y_train_m)
        last_oos_ind = first_oos_ind + len(y_test) - 1
        y_pred_out = fin_model.predict(start=first_oos_ind, end=last_oos_ind, exog_oos=test_data)
        y_pred_out.reset_index(drop=True, inplace=True)

        MAE_test = np.nanmean(abs(y_pred_out - y_test))
        MSE_test = mean_squared_error(y_pred_out, y_test)
        RMSE_test = math.sqrt(MSE_test)
        RAE_test = rae(actual=y_test, predicted = y_pred_out)
        RRSE_test = rrse(actual=y_test, predicted = y_pred_out)



        if orders_of_integ[target] > 0:

            # Calculating test scores in the original scale
            if stationarity_method == 0:
                # y_pred_out_destat = y_pred_out.copy()
                # y_pred_out_destat.loc[-1] = y_test_non_stat.iloc[max_sel_lag]
                # y_pred_out_destat.index = y_pred_out_destat.index + 1
                # y_pred_out_destat = y_pred_out_destat.sort_index()
                # y_pred_out_destat = y_pred_out_destat.cumsum()

                # y_test_non_stat_destat = y_test_non_stat.copy()
                # y_test_non_stat_destat = y_test_non_stat_destat.iloc[max_sel_lag:]
                # y_test_non_stat_destat.reset_index(drop=True, inplace=True)
                
                # MAE_test_destat = np.nanmean(abs(y_pred_out_destat - y_test_non_stat_destat))
                y_pred_out_destat = y_pred_out.copy()
                y_test_non_stat_destat = y_test_non_stat.copy()
                y_test_non_stat_destat = y_test_non_stat_destat.iloc[max_sel_lag:]
                y_test_non_stat_destat.reset_index(drop=True, inplace=True)

                y_pred_out_destat = y_test_non_stat_destat.iloc[:-1] + y_pred_out

                y_test_non_stat_destat = y_test_non_stat_destat.iloc[1:]
                y_test_non_stat_destat.reset_index(drop=True, inplace=True)

                MAE_test_destat = np.nanmean(abs(y_pred_out_destat - y_test_non_stat_destat))
                RAE_test = rae(actual=y_test_non_stat_destat, predicted = y_pred_out_destat)

            elif stationarity_method == 1:
                y_test_logged = np.log(y_test_non_stat)
                y_pred_out_destat = y_pred_out.copy()
                y_pred_out_destat.loc[-1] = y_test_logged.iloc[max_sel_lag]
                y_pred_out_destat.index = y_pred_out_destat.index + 1
                y_pred_out_destat = y_pred_out_destat.sort_index()
                y_pred_out_destat = np.exp(y_pred_out_destat.cumsum())

                y_test_non_stat_destat = y_test_logged.copy()
                y_test_non_stat_destat = y_test_non_stat_destat.iloc[max_sel_lag:]
                y_test_non_stat_destat.reset_index(drop=True, inplace=True)
                y_test_non_stat_destat = np.exp(y_test_non_stat_destat)

                MAE_test_destat = np.nanmean(abs(y_pred_out_destat - y_test_non_stat_destat))

            MAE = {"train": MAE_train_destat, "test": MAE_test_destat}
        else:
            MAE = {"train": MAE_train, "test": MAE_test}
            y_train_m_destat = y_train_m
            y_test_non_stat_destat = y_test




        
        MAE_stat = {"train": MAE_train, "test": MAE_test}
        MAE_ = {"Original": MAE, "Stationary": MAE_stat}
        logging.info("Check")






        #Non-stationary dataset with selected features only, to be used in evoML
        selected_vars = list(feature_n_dfs_merge.columns)
        names_without_L = []
        for var in selected_vars:
            name_without_L = var.split(".")[0]
            if name_without_L != "const" and (name_without_L in names_without_L)==False:
                names_without_L.append(name_without_L)
            else:
                continue

        only_seleceted_vars_df = df[names_without_L]
        selected_dfs_lags = []
        for var in names_without_L:
            columns = []
            for i in list(range(1, max_lag+1)):
                columns.append(var + ".L"+str(i))

            selcted_var_lag_df = pd.DataFrame(columns=columns)
            for i in list(range(max_lag)):
                selcted_var_lag_df[columns[i]] = only_seleceted_vars_df[var].shift(i+1)
            
            selected_dfs_lags.append(selcted_var_lag_df)
        selected_dfs_lags = pd.concat(selected_dfs_lags, axis=1)
        selected_features_df = selected_dfs_lags[selected_vars]
        
        export_df = pd.concat([df[["Date", target]], selected_features_df], axis=1)
        export_df = export_df.iloc[max_lag:,:]
        export_df.reset_index(inplace=True, drop=True)

        export_df_evoml = pd.concat([df[["Date", target]], selected_features_df], axis=1)
        export_df_evoml[target] = export_df_evoml[target].shift(1)
        export_df_evoml.pop(target+".L1")
        export_df_evoml["Date"] = export_df_evoml["Date"].shift(1)
        export_df_evoml = export_df_evoml.iloc[max_lag:,:]
        export_df_evoml.reset_index(inplace=True, drop=True)


        end = time.time()
        time_to_run = end - start


        #Saving the test and train metrics 
        destat_data = {"y_train": y_train_m_destat, "y_test": y_test_non_stat_destat,
                        "stationarity_method": stationarity_method,
                        "y_integ_order": orders_of_integ[target]}
        my_metrics_test = {"MAE":[MAE_test], "RMSE":[RMSE_test], "RAE":[RAE_test], "RRSE":[RRSE_test]}
        my_metrics_train = {"MAE":[MAE_train], "RMSE":[RMSE_train], "RAE":[RAE_train], "RRSE":[RRSE_train]}
        my_metrics_CV = {"MAE": [MAE_CV]}
        my_metrics = {"train":my_metrics_train, "CV":my_metrics_CV, "test":my_metrics_test}

        Model_Data = Sun_Model(fin_model, fin_model.summary(), aug_models, MAE_,
                                y_train_m, feature_n_dfs_merge,
                                y_test, test_data,
                                y_pred_out, destat_data,
                                my_metrics,
                                export_df,
                                export_df_evoml,
                                time_to_run,
                                selected_vars)




        return Model_Data
    except ValueError:
        logging.error("Can not reject that the target variable 'reverse causes' independent features.")