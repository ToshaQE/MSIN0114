import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path  
from MAIN_STAGE_ONE import stage_one
from MAIN_STAGE_ONE import stage_one_ns






#Reading in the data
df_aapl = pd.read_csv("df_aaple.csv")
# Truncating the dataw
aapl_long = df_aapl.iloc[:,:22]

Model_Data = stage_one_ns(df=aapl_long, target="Close", max_lag=20, stationarity_method = 0, test_size=0.2)



filepath = Path('Apple/Data/Selected_dfs/apple_stage_one_5.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
Model_Data.export_data.to_csv(filepath, index=False)

filepath = Path('Apple/Data/evoML/apple_stage_one_5_evoML.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
Model_Data.export_data_evoML.to_csv(filepath, index=False)

print(Model_Data.summary)
print(f'MAE on the original scale is: \n{Model_Data.MAE["Original"]}\n\n')
print(f'MAE on the stationaries scale is: \n{Model_Data.MAE["Stationary"]}')
print(Model_Data.my_metrics["test"])
print(f'MAE CV is: \n{Model_Data.my_metrics["CV"]}')
print(Model_Data.selected_features)

filename = 'Sun_Model_Data_applle'
outfile = open(filename,'wb')
pickle.dump(Model_Data,outfile)
outfile.close()

# apple_long_evoML = pd.concat([Model_Data.y_train])

# {'train': 1.2125139241871459, 'test': 1.199242993289765}











