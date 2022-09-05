from typing import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path



#Reading in and formatting the data for the visualistion
# filepath = Path('Apple/Tables/evoML_apple_stage_one.xlsx')
filepath = Path('Apple/Tables/evoML_apple_stage_two.xlsx')
# filepath = Path('Apple/Tables/attempt.xlsx')

filepath.parent.mkdir(parents=True, exist_ok=True)
results = pd.read_excel(filepath, sheet_name="All")

results = pd.melt(results, id_vars="Model", var_name="Method", value_name="MAE")
vis1_yticks = [0, 1, 2, 3, 4, 5]
#Plot settings
sns.set_theme()
sns.set_palette("tab10")
vis1 = sns.catplot(data=results, x="Model", y="MAE", hue="Method", kind="bar", legend=False,
                    height=6, aspect=6/8)
vis1.set_axis_labels("", "MAE", fontweight = 'bold', fontsize=14)
# vis1.set_yticklabels(vis1_yticks, fontsize = 12)
vis1.set_xticklabels(fontsize = 11, fontweight = 'bold')
plt.yticks(np.arange(0, 11, 1))
plt.legend(loc="upper left")
plt.show()


#Reading in the data and formatting it for visualisation in relative terms
results_pc = pd.read_excel(filepath, sheet_name="All")

# In each row, divide each method's score by the score of the 'Original'
n_rows = range(len(results_pc))
methods = list(results_pc.columns)[:-1]

for row in n_rows:
    counter = 0
    score_original = results_pc.iloc[row, 1]
    for method in methods:
        results_pc.iloc[row, counter] = (results_pc.iloc[row, counter]/score_original)*100
        counter += 1


results_pc = pd.melt(results_pc, id_vars="Model", var_name="Method", value_name="MAE")

#Plot settings
sns.set_theme()
sns.set_palette("tab10")
vis2 = sns.catplot(data=results_pc, x="Model", y="MAE", hue="Method", kind="bar", legend=False,
                    height=6, aspect=6/8)
vis2.set_axis_labels("", 'MAE (% Relative to the "Original" score)', fontweight = 'bold', fontsize=14)
vis1.set_yticklabels(fontsize = 13)
vis2.set_xticklabels(fontsize = 11, fontweight = 'bold')
plt.yticks(np.arange(0, 300, 20))
plt.legend(loc="upper left")
plt.show()


print("Stop")
