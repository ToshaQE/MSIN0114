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

from MAIN_STAGE_TWO import stage_two_reverse


# # Scaling values in the feature set
# scaling = MinMaxScaler(feature_range=(0,1)).fit(sun_x_train)
# X_train = scaling.transform(sun_x_train)
# X_test = scaling.transform(sun_x_test)






# Unpickling the data

infile = open("Sun_Model_Data_apple",'rb')
Model_Data_apple = pickle.load(infile)
infile.close()


infile = open("Sun_Model_Data_chickenpox",'rb')
Model_Data_chickenpox = pickle.load(infile)
infile.close()





models_apple = [LGBMRegressor]
model_names_apple = ['Light GBM']

grids_apple = [
 {
 'random_state': [0],
 'n_jobs': [-1],
 }
]


models_chickenpox = [RandomForestRegressor]
model_names_chickenpox = ['Random Forest']

grids_chickenpox = [
 {
 'random_state': 0,
 'n_jobs': -1,
 }
]




FRUFS_Data_apple = stage_two_reverse(Model_Data_apple, models_apple, model_names_apple, grids_apple)
FRUFS_Data_chickenpox = stage_two_reverse(Model_Data_chickenpox, models_chickenpox, model_names_chickenpox, grids_chickenpox)




# Creating plots
LGBM_Vis = FRUFS_Data_apple["Light GBM"]["Visualisation"]
RF_Vis = FRUFS_Data_chickenpox["Random Forest"]["Visualisation"]


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8.5, 4.25))


# sns.set_theme()
sns.set_palette("tab10")
sns.despine()

plt.subplot(1,2,1)
ax4 = sns.lineplot(x='Feature %', y='value', hue='variable',
            data=LGBM_Vis, marker="o", linewidth = 2.5, markersize=8, legend=False)
ax4.set(xlabel='% of Features Dropped', ylabel='% Change over the Original Train/CV Error')
plt.title("Apple", fontweight = 'bold')
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)



plt.subplot(1,2,2)
ax2 = sns.lineplot(x='Feature %', y='value', hue='variable',
            data=RF_Vis, marker="o", linewidth = 2.5, markersize=8, legend=False)
ax2.set(xlabel='% of Features Dropped', ylabel='% Change over the Original Train/CV Error')
plt.title("Chickenpox", fontweight = 'bold')
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)








fig.text(0.5, 0.01, '% of Features Dropped', ha='center', fontsize=12.9, fontweight = 'bold')
fig.text(0.04, 0.5, '% Change over the Original Train/CV Error', va='center', rotation='vertical', fontsize=12.9, fontweight = 'bold')
# plt.legend()
plt.show()
fig.savefig("./Apple/Figures/estimators_reverse.png")
fig.savefig("./Chickenpox/Figures/estimators_reverse.png")



print("Stop")

