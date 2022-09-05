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
from MAIN_STAGE_TWO import stage_two
from lag_functions import lag_n_times, lag_n_times_evoML
from pathlib import Path 


# Unpickling the data
infile = open("Sun_Model_Data_chickenpox",'rb')
Model_Data = pickle.load(infile)
infile.close()

models = [LinearRegression, RandomForestRegressor, XGBRegressor, LGBMRegressor]
model_names = ['LinearRegression', 'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor']

grids = [{
 'fit_intercept': [True]
 }
,

 {
 'random_state': 42,
 'n_jobs': -1
 },
 {
 'random_state': 42,
 'n_jobs': -1
 },
 {
 'random_state': [42],
 'n_jobs': [-1],
 }
]



FRUFS_Data, Rankings_Table, CV_Change_Table, Test_Change_Table = stage_two(Model_Data, models, model_names, grids)

# Exporting the tables
Rankings_Table.to_csv("./Chickenpox/Tables/Rankings_Table_chickenpox.csv", index=False)
CV_Change_Table.to_csv("./Chickenpox/Tables/CV_Change_Table_chickenpox.csv", index=False)
Test_Change_Table.to_csv("./Chickenpox/Tables/Test_Change_Table_chickenpox.csv", index=False)




# Creating plots
LinearR_Vis = FRUFS_Data["LinearRegression"]["Visualisation"]
RF_Vis = FRUFS_Data["RandomForestRegressor"]["Visualisation"]
XGB_Vis = FRUFS_Data["XGBRegressor"]["Visualisation"]
LGBM_Vis = FRUFS_Data["LGBMRegressor"]["Visualisation"]


fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(8.5, 8.5))


sns.set_palette("tab10")
sns.despine()


plt.subplot(2,2,1)


ax1 = sns.lineplot(x='Feature %', y='value', hue='variable',
            data=LinearR_Vis, marker="o", linewidth = 2.5, markersize=8)
ax1.set(xlabel='% of Features Dropped', ylabel='% Change over the Original Train/CV Error')
plt.title("Linear Regression", fontweight = 'bold')
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

plt.subplot(2,2,2)
ax2 = sns.lineplot(x='Feature %', y='value', hue='variable',
            data=RF_Vis, marker="o", linewidth = 2.5, markersize=8, legend=False)
ax2.set(xlabel='% of Features Dropped', ylabel='% Change over the Original Train/CV Error')
plt.title("Random Forest", fontweight = 'bold')
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(2,2,3)
ax3 = sns.lineplot(x='Feature %', y='value', hue='variable',
            data=XGB_Vis, marker="o", linewidth = 2.5, markersize=8, legend=False)
ax3.set(xlabel='% of Features Dropped', ylabel='% Change over the Original Train/CV Error')
plt.title("XGBoost", fontweight = 'bold')
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(2,2,4)
ax4 = sns.lineplot(x='Feature %', y='value', hue='variable',
            data=LGBM_Vis, marker="o", linewidth = 2.5, markersize=8, legend=False)
ax4.set(xlabel='% of Features Dropped', ylabel='% Change over the Original Train/CV Error')
plt.title("Light GBM", fontweight = 'bold')
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)




fig.text(0.5, 0.04, '% of Features Dropped', ha='center', fontsize=15, fontweight = 'bold')
fig.text(0.04, 0.5, '% Change over the Original Train/CV Error', va='center', rotation='vertical', fontsize=15, fontweight = 'bold')
# plt.legend()
plt.show()

filepath = Path('Chickenpox/Figures/estimators_chickenpox.png')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
fig.savefig(filepath)



# Features selected
features_selected_stage_2_chickenpox = FRUFS_Data["RandomForestRegressor"]["Selected Features"]
filename = 'features_selected_stage_2_chickenpox'
outfile = open(filename,'wb')
pickle.dump(features_selected_stage_2_chickenpox,outfile)
outfile.close()


#Reading in the data
chickenpox_df = pd.read_csv("hungary_chickenpox1.csv")


chickenpox_stage_two_3 = lag_n_times(chickenpox_df, 15, "BUDAPEST", features_selected_stage_2_chickenpox)
filepath = Path('Chickenpox/Data/Selected_dfs/chickenpox_stage_two_3.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
chickenpox_stage_two_3.to_csv(filepath, index=False)

chickenpox_stage_two_3_evoML = lag_n_times_evoML(chickenpox_df, 15, "BUDAPEST", features_selected_stage_2_chickenpox)
filepath = Path('Chickenpox/Data/evoML/chickenpox_stage_two_3_evoML.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
chickenpox_stage_two_3_evoML.to_csv(filepath, index=False)



print(features_selected_stage_2_chickenpox)
print(Rankings_Table)
print(CV_Change_Table)
print(Test_Change_Table)

print(FRUFS_Data['RandomForestRegressor']["Time to Run"])
print("Stop")

