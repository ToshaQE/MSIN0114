import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import grangercausalitytests
import pickle
from sklearn.model_selection import train_test_split
import re

from FRUFS import FRUFS
import matplotlib.pyplot as plt
import optuna
import joblib, gc
import lightgbm as lgb
import seaborn as sns

from sklearn.datasets import make_regression
from scipy.stats import pearsonr
from tqdm.notebook import trange, tqdm
from FRUFS import FRUFS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statistics
from sklearn.model_selection import TimeSeriesSplit
from sigfig import round
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
import time

def stage_two(Model_Data, models, model_names, grids):

    counter = 0

    # Reading in the data
    sun_x_train = Model_Data.train_x
    sun_y_train = Model_Data.train_y
    sun_y_test = Model_Data.test_y
    sun_x_test = Model_Data.test_x
    sun_MAE_train = Model_Data.MAE["Stationary"]["train"]
    sun_MAE_test = Model_Data.MAE["Stationary"]["test"]
    sun_MAE_CV = Model_Data.my_metrics["CV"]["MAE"][0]
    sun_fin_model = Model_Data.fin_model
    sun_model_params = list(sun_fin_model.params.index)
    total_n_features = len(list(sun_x_train.columns))

    FRUFS_Data = {}

    for name in model_names:
        FRUFS_Data[name] = {}


    for model in models:

        params = grids[counter]

        # Defining FRUFS model with k=maximum features
        model_frufs_generated = FRUFS(
                    model_r = model().set_params(**params),
                    k = total_n_features,
                    n_jobs = -1
                )



        # Recording the start time
        start = time.time()

        # fit_transform returns data ranked in the order of importance
        pruned_df = model_frufs_generated.fit_transform(sun_x_train)
        FRUFS_Loop = {"MAE Train":[], "MAE CV":[], "Feature %":[], "Feature Dropped":[]}





        #Dropping the least important feature, calulcating the train and CV errors, and reestimating the weights again
        for n in list(range(total_n_features, 0, -1)):

            if n != total_n_features and n != 1:
                column_names = list(pruned_df_cut.columns)
                column_dropped = column_names[n]
                print(n)
                pruned_df_cut = model_frufs_generated.fit_transform(pruned_df_cut.iloc[:,:n])
            elif n == 1:
                column_names = list(pruned_df_cut.columns)
                column_dropped = column_names[n]
                pruned_df_cut = pruned_df_cut.iloc[:,:n]

            else:
                column_dropped = "None"
                pruned_df_cut =  pruned_df
                column_names = list(pruned_df_cut.columns)








            # Training the model on the features selected by FRUFS
            # Check whether the Sun Model has a constant or not
            if "const" in sun_model_params:
                frufs_model = AutoReg(sun_y_train, lags=0, exog=pruned_df_cut).fit()
            else:
                frufs_model = AutoReg(sun_y_train, lags=0, exog=pruned_df_cut, trend="n").fit()








            # Calculating MAE train (.sqeeze is reuired when sun_y is saved as a DF and not series)
            MAE_train = np.nanmean(abs(frufs_model.predict() - sun_y_train.squeeze()))







            # Calculating CV error
            tscv = TimeSeriesSplit(n_splits=10)
            MAE_CV_list = []
            for train_index, test_index in tscv.split(sun_y_train):
                X_CV_train, X_CV_test = pruned_df_cut.iloc[train_index], pruned_df_cut.iloc[test_index]
                y_CV_train, y_CV_test = sun_y_train.iloc[train_index], sun_y_train.iloc[test_index]

                model_CV = AutoReg(y_CV_train, lags=0, exog=X_CV_train).fit()
                y_CV_pred = model_CV.predict(start=test_index[0], end=test_index[-1], exog_oos = X_CV_test)
                MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
                MAE_CV_list.append(MAE_CV)
            
            MAE_CV = statistics.mean(MAE_CV_list)








            print(f"MAE Train = {MAE_train}; MAE_CV = {MAE_CV}")







            FRUFS_Loop["MAE Train"].append(MAE_train)
            FRUFS_Loop["MAE CV"].append(MAE_CV)
            FRUFS_Loop["Feature %"].append((n/total_n_features)*100)
            FRUFS_Loop["Feature Dropped"].append(column_dropped)
        
        # Recording the start time
        end = time.time()
        time_to_run = end - start

        FRUFS_Loop["MAE Train % Change"] = ((FRUFS_Loop["MAE Train"]/sun_MAE_train) - 1)*100
        FRUFS_Loop["MAE CV % Change"] = ((FRUFS_Loop["MAE CV"]/sun_MAE_CV) - 1)*100
        FRUFS_Loop = pd.DataFrame.from_dict(FRUFS_Loop)
        FRUFS_Loop.rename(columns = {'MAE Train % Change':'Train', 'MAE CV % Change':'CV'}, inplace = True)
        last_feature = column_names[0]

        # Storing the data for table of ranked features
        # FRUFS_Data[model_names[counter]]["Ranked Features"] = list(FRUFS_Loop["Feature Dropped"]).append(last_feature)



        #Estimating the test score imporovement on the subset where CV error is minimised
        features_dropped = list(FRUFS_Loop["Feature Dropped"])
        test_changes = list(FRUFS_Loop["CV"])
        feature_pcs = abs(100 - FRUFS_Loop["Feature %"])
        #Index where the CV error is minimised
        min_error_ind = test_changes.index(min(test_changes))
        features_selected = features_dropped[min_error_ind+1:]
        last_remaining_feature = column_names[0]
        features_selected.append(last_remaining_feature)

        optimal_x_train = sun_x_train[features_selected]

        if 'const' in sun_model_params:
            model = LinearRegression().fit(optimal_x_train, sun_y_train)  
        else:
            model = LinearRegression(fit_intercept=False).fit(optimal_x_train, sun_y_train)

        #Calculating the error on the test set
        X_selected_test = sun_x_test[features_selected]
        pred_out = model.predict(X_selected_test)
        MAE_test = mean_absolute_error(sun_y_test, pred_out)
        #Calculating the % change in the error compared to the original test error
        MAE_change = (MAE_test/sun_MAE_test - 1)*100
        #Storing the change
        FRUFS_Data[model_names[counter]]["CV MAE Change"] = test_changes[min_error_ind]
        FRUFS_Data[model_names[counter]]["Test MAE Change"] = MAE_change



        # Formatting the data for visualistions
        FRUFS_Long = pd.melt(FRUFS_Loop, ['Feature %'])
        FRUFS_Vis = FRUFS_Long.loc[FRUFS_Long['variable'].isin(["Train", "CV"])]
        FRUFS_Vis["Feature %"] = abs(100 - FRUFS_Vis["Feature %"])
        FRUFS_Long = pd.melt(FRUFS_Loop, ['Feature %'])
        FRUFS_Vis = FRUFS_Long.loc[FRUFS_Long['variable'].isin(["Train", "CV"])]
        FRUFS_Vis["Feature %"] = abs(100 - FRUFS_Vis["Feature %"])
        # Storing the visulisaton data
        FRUFS_Data[model_names[counter]]["Visualisation"] = FRUFS_Vis   


        # Ranking the features is the same as the reverse order of them dropped
        # Getting rid of "None"
        features_dropped = features_dropped[1:]
        features_dropped.append(last_feature)
        # Ranking the features is the same as the reverse order of them dropped
        features_ranked = features_dropped[::-1]
        FRUFS_Data[model_names[counter]]["Ranked Features"] = features_ranked


        FRUFS_Data[model_names[counter]]["Selected Features"] = features_selected

        FRUFS_Data[model_names[counter]]["% of Features Dropped"] = feature_pcs[min_error_ind]

        FRUFS_Data[model_names[counter]]["Time to Run"] = time_to_run

        counter += 1





    #Creating dataframe with rankings of features compiled by each model
    rankings = list(range(1, total_n_features + 1))
    feature_pcs = list(feature_pcs[1:])
    feature_pcs.append("N/A")
    feature_pcs = feature_pcs[::-1]

    Table_Data = {"Ranking": rankings, '% of Features Dropped': feature_pcs}

    for name in model_names:
        ranked_features =  FRUFS_Data[name]["Ranked Features"]
        Table_Data[name] = ranked_features

    Rankings_Table = pd.DataFrame.from_dict(Table_Data)






    #Creating dataframe with % change for each method
    CV_Change_Data = {'Index': range(0, len(model_names)), "Estimator":[], "% Change in CV Error":[], "% of Features Dropped":[]}

    for name in model_names:
        CV_Change_Data["Estimator"].append(name)
        CV_Change_Data["% Change in CV Error"].append(FRUFS_Data[name]["CV MAE Change"])
        CV_Change_Data['% of Features Dropped'].append(FRUFS_Data[name]["% of Features Dropped"])
    

    CV_Change_Table = pd.DataFrame.from_dict(CV_Change_Data)

    
    #Creating dataframe with % change for each method
    Test_Change_Data = {'Index': range(0, len(model_names)), "Estimator":[], "% Change in Test Error":[]}

    for name in model_names:
        Test_Change_Data["Estimator"].append(name)
        Test_Change_Data["% Change in Test Error"].append(FRUFS_Data[name]["Test MAE Change"])
    

    Test_Change_Table = pd.DataFrame.from_dict(Test_Change_Data)



    return FRUFS_Data, Rankings_Table, CV_Change_Table, Test_Change_Table


def stage_two_reverse(Model_Data, models, model_names, grids):

    counter = 0

    # Reading in the data
    sun_x_train = Model_Data.train_x
    sun_y_train = Model_Data.train_y
    sun_y_test = Model_Data.test_y
    sun_x_test = Model_Data.test_x
    sun_MAE_train = Model_Data.MAE["Stationary"]["train"]
    sun_MAE_test = Model_Data.MAE["Stationary"]["test"]
    sun_MAE_CV = Model_Data.my_metrics["CV"]["MAE"][0]
    sun_fin_model = Model_Data.fin_model
    sun_model_params = list(sun_fin_model.params.index)
    total_n_features = len(list(sun_x_train.columns))

    FRUFS_Data = {}

    for name in model_names:
        FRUFS_Data[name] = {}


    for model in models:

        params = grids[counter]

        # Defining FRUFS model with k=maximum features
        model_frufs_generated = FRUFS(
                    model_r = model().set_params(**params),
                    k = total_n_features,
                    n_jobs = -1
                )


        # fit_transform returns data ranked in the order of importance
        pruned_df = model_frufs_generated.fit_transform(sun_x_train)
        FRUFS_Loop = {"MAE Train":[], "MAE CV":[], "Feature %":[], "Feature Dropped":[]}





        #Dropping the most important feature, calulcating the train and CV errors, and reestimating the weights again
        for n in list(range(total_n_features, 0, -1)):
            if n != total_n_features and n != 1:
                pruned_df_cut = pruned_df_cut[pruned_df_cut.columns[::-1]]
                column_names = list(pruned_df_cut.columns)
                column_dropped = column_names[n]
                print(n)
                pruned_df_cut = model_frufs_generated.fit_transform(pruned_df_cut.iloc[:,:n])
            elif n == 1:
                pruned_df_cut = pruned_df_cut[pruned_df_cut.columns[::-1]]
                column_names = list(pruned_df_cut.columns)
                column_dropped = column_names[n]
                pruned_df_cut = pruned_df_cut.iloc[:,:n]

            else:
                column_dropped = "None"
                pruned_df_cut =  pruned_df








            # Training the model on the features selected by FRUFS
            # Check whether the Sun Model has a constant or not
            if "const" in sun_model_params:
                frufs_model = AutoReg(sun_y_train, lags=0, exog=pruned_df_cut).fit()
            else:
                frufs_model = AutoReg(sun_y_train, lags=0, exog=pruned_df_cut, trend="n").fit()








            # Calculating MAE train (.sqeeze is reuired when sun_y is saved as a DF and not series)
            MAE_train = np.nanmean(abs(frufs_model.predict() - sun_y_train.squeeze()))







            # Calculating CV error
            tscv = TimeSeriesSplit(n_splits=10)
            MAE_CV_list = []
            for train_index, test_index in tscv.split(sun_y_train):
                X_CV_train, X_CV_test = pruned_df_cut.iloc[train_index], pruned_df_cut.iloc[test_index]
                y_CV_train, y_CV_test = sun_y_train.iloc[train_index], sun_y_train.iloc[test_index]

                model_CV = AutoReg(y_CV_train, lags=0, exog=X_CV_train).fit()
                y_CV_pred = model_CV.predict(start=test_index[0], end=test_index[-1], exog_oos = X_CV_test)
                MAE_CV = np.nanmean(abs(y_CV_pred - y_CV_test))
                MAE_CV_list.append(MAE_CV)
            
            MAE_CV = statistics.mean(MAE_CV_list)








            print(f"MAE Train = {MAE_train}; MAE_CV = {MAE_CV}")







            FRUFS_Loop["MAE Train"].append(MAE_train)
            FRUFS_Loop["MAE CV"].append(MAE_CV)
            FRUFS_Loop["Feature %"].append((n/total_n_features)*100)
            FRUFS_Loop["Feature Dropped"].append(column_dropped)


        FRUFS_Loop["MAE Train % Change"] = ((FRUFS_Loop["MAE Train"]/sun_MAE_train) - 1)*100
        FRUFS_Loop["MAE CV % Change"] = ((FRUFS_Loop["MAE CV"]/sun_MAE_CV) - 1)*100
        FRUFS_Loop = pd.DataFrame.from_dict(FRUFS_Loop)
        FRUFS_Loop.rename(columns = {'MAE Train % Change':'Train', 'MAE CV % Change':'CV'}, inplace = True)
        last_feature = column_names[0]

        # Storing the data for table of ranked features
        # FRUFS_Data[model_names[counter]]["Ranked Features"] = list(FRUFS_Loop["Feature Dropped"]).append(last_feature)



        #Estimating the test score imporovement on the subset where CV error is minimised
        features_dropped = list(FRUFS_Loop["Feature Dropped"])
        test_changes = list(FRUFS_Loop["CV"])
        feature_pcs = abs(100 - FRUFS_Loop["Feature %"])
        #Index where the CV error is minimised
        min_error_ind = test_changes.index(min(test_changes))
        features_selected = features_dropped[min_error_ind+1:]
        last_remaining_feature = column_names[0]
        features_selected.append(last_remaining_feature)

        optimal_x_train = sun_x_train[features_selected]

        if 'const' in sun_model_params:
            model = LinearRegression().fit(optimal_x_train, sun_y_train)  
        else:
            model = LinearRegression(fit_intercept=False).fit(optimal_x_train, sun_y_train)

        #Calculating the error on the test set
        X_selected_test = sun_x_test[features_selected]
        pred_out = model.predict(X_selected_test)
        MAE_test = mean_absolute_error(sun_y_test, pred_out)
        #Calculating the % change in the error compared to the original test error
        MAE_change = (MAE_test/sun_MAE_test - 1)*100
        #Storing the change
        FRUFS_Data[model_names[counter]]["CV MAE Change"] = test_changes[min_error_ind]
        FRUFS_Data[model_names[counter]]["Test MAE Change"] = MAE_change



        # Formatting the data for visualistions
        FRUFS_Long = pd.melt(FRUFS_Loop, ['Feature %'])
        FRUFS_Vis = FRUFS_Long.loc[FRUFS_Long['variable'].isin(["Train", "CV"])]
        FRUFS_Vis["Feature %"] = abs(100 - FRUFS_Vis["Feature %"])
        FRUFS_Long = pd.melt(FRUFS_Loop, ['Feature %'])
        FRUFS_Vis = FRUFS_Long.loc[FRUFS_Long['variable'].isin(["Train", "CV"])]
        FRUFS_Vis["Feature %"] = abs(100 - FRUFS_Vis["Feature %"])
        # Storing the visulisaton data
        FRUFS_Data[model_names[counter]]["Visualisation"] = FRUFS_Vis   


        # Ranking the features is the same as the reverse order of them dropped
        # Getting rid of "None"
        features_dropped = features_dropped[1:]
        features_dropped.append(last_feature)
        # Ranking the features is the same as the reverse order of them dropped
        features_ranked = features_dropped[::-1]
        FRUFS_Data[model_names[counter]]["Ranked Features"] = features_ranked


        FRUFS_Data[model_names[counter]]["Selected Features"] = features_selected

        FRUFS_Data[model_names[counter]]["% of Features Dropped"] = feature_pcs[min_error_ind]

        counter += 1





    #Creating dataframe with rankings of features compiled by each model
    rankings = list(range(1, total_n_features + 1))
    feature_pcs = list(feature_pcs[1:])
    feature_pcs.append("N/A")
    feature_pcs = feature_pcs[::-1]

    Table_Data = {"Ranking": rankings, '% of Features Dropped': feature_pcs}

    for name in model_names:
        ranked_features =  FRUFS_Data[name]["Ranked Features"]
        Table_Data[name] = ranked_features

    Rankings_Table = pd.DataFrame.from_dict(Table_Data)






    #Creating dataframe with % change for each method
    CV_Change_Data = {'Index': range(0, len(model_names)), "Estimator":[], "% Change in CV Error":[], "% of Features Dropped":[]}

    for name in model_names:
        CV_Change_Data["Estimator"].append(name)
        CV_Change_Data["% Change in CV Error"].append(FRUFS_Data[name]["CV MAE Change"])
        CV_Change_Data['% of Features Dropped'].append(FRUFS_Data[name]["% of Features Dropped"])
    

    CV_Change_Table = pd.DataFrame.from_dict(CV_Change_Data)

    
    #Creating dataframe with % change for each method
    Test_Change_Data = {'Index': range(0, len(model_names)), "Estimator":[], "% Change in Test Error":[]}

    for name in model_names:
        Test_Change_Data["Estimator"].append(name)
        Test_Change_Data["% Change in Test Error"].append(FRUFS_Data[name]["Test MAE Change"])
    

    Test_Change_Table = pd.DataFrame.from_dict(Test_Change_Data)



    return FRUFS_Data