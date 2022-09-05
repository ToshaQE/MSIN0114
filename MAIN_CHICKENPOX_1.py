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
from pathlib import Path  
from Sun_Model_Class import Sun_Model
import pmdarima as pmd
from my_metrics import rae, rrse
from lag_functions import lag_n_times
import time
from MAIN_STAGE_ONE import stage_one
from MAIN_STAGE_ONE import stage_one_ns






#Reading in the data
chickenpox_df = pd.read_csv("hungary_chickenpox1.csv")

Model_Data = stage_one(df=chickenpox_df, target="BUDAPEST", max_lag=15, stationarity_method = 0, test_size=0.2)



filepath = Path('Chickenpox/Data/Selected_dfs/chickenpox_stage_one_13.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
Model_Data.export_data.to_csv(filepath, index=False)

filepath = Path('Chickenpox/Data/evoML/chickenpox_stage_one_13_evoML.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
Model_Data.export_data_evoML.to_csv(filepath, index=False)


print(Model_Data.summary)
print(f'MAE on the original scale is: \n{Model_Data.MAE["Original"]}\n\n')
print(f'MAE on the stationaries scale is: \n{Model_Data.MAE["Stationary"]}')
print(Model_Data.my_metrics["test"])
print(f'MAE CV is: \n{Model_Data.my_metrics["CV"]}')


filename = 'Sun_Model_Data_chickenpox'
outfile = open(filename,'wb')
pickle.dump(Model_Data,outfile)
outfile.close()















