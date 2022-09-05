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
from lag_functions import lag_n_times, lag_n_times_evoML
import time


from Classical_FS import RFE_FS, BE, FC
from pathlib import Path  




# Reading in the data
chickenpox_df = pd.read_csv("hungary_chickenpox1.csv")

infile = open("Sun_Model_Data_chickenpox",'rb')
Model_Data = pickle.load(infile)
infile.close()
stage_one_selected = Model_Data.selected_features

#Defining dictionaries to store the names of selected variables
selected_unique_features_table = {"RFE":[], "BE":[], "FC":[]}
selected_features_table = {"Causality":stage_one_selected, "RFE":[], "BE":[], "FC":[]}


# Running RFE-based feature selection algorithm and saving the results
selected_unique, selected_all, MAE, MAE_all, selected_df_evoML, selected_df, time_to_run = RFE_FS(df=chickenpox_df, target="BUDAPEST",
    test_size=0.2, n_features=13, n_features_2=28, max_lag=15)


selected_unique_features_table["RFE"] = selected_unique
selected_features_table["RFE"] = selected_all

filepath = Path('Chickenpox/Data/evoML/chickenpox_RFE_LinearSVR_13_evoML.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
selected_df_evoML.to_csv(filepath, index=False)

filepath = Path('Chickenpox/Data/Selected_dfs/chickenpox_RFE_LinearSVR_13.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
selected_df.to_csv(filepath, index=False)





# Running BE-based feature selection algorithm and saving the results
selected_unique, selected_all, MAE, MAE_all, selected_df_evoML, selected_df, time_to_run = BE(df=chickenpox_df, target="BUDAPEST",
    test_size=0.2, n_features=13, n_features_2=28, max_lag=15)

selected_unique_features_table["BE"] = selected_unique
selected_features_table["BE"] = selected_all

filepath = Path('Chickenpox/Data/evoML/chickenpox_BE_evoML.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
selected_df_evoML.to_csv(filepath, index=False)

filepath = Path('Chickenpox/Data/Selected_dfs/chickenpox_BE_13.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
selected_df.to_csv(filepath, index=False)






# Running FC-based feature selection algorithm and saving the results
selected_unique, selected_all, MAE, MAE_all, selected_df_evoML, selected_df, time_to_run = FC(df=chickenpox_df, target="BUDAPEST",
    test_size=0.2, n_features=13, n_features_2=28, max_lag=15)

selected_unique_features_table["FC"] = selected_unique
selected_features_table["FC"] = selected_all

filepath = Path('Chickenpox/Data/evoML/chickenpox_FC_13_evoML.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
selected_df_evoML.to_csv(filepath, index=False)

filepath = Path('Chickenpox/Data/Selected_dfs/chickenpox_FC_13.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
selected_df.to_csv(filepath, index=False)



# Saving data for the tables
selected_unique_features_table = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in selected_unique_features_table.items() ]))
selected_features_table = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in selected_features_table.items() ]))

filepath = Path('Chickenpox/Tables/chickenpox_13_unique.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
selected_unique_features_table.to_csv(filepath, index=False)

filepath = Path('Chickenpox/Tables/chickenpox_13_all.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
selected_features_table.to_csv(filepath, index=False)

print("Stop")