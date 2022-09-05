import pandas as pd
import numpy as np

def lag_n_times(df, n, target, selected_f_names):
    #Non-stationary dataset with selected features only, to be used in evoML
    # selected_vars = list(feature_n_dfs_merge.columns)
    # names_without_L = []
    # for var in selected_vars:
    #     name_without_L = var.split(".")[0]
    #     if name_without_L != "const" and (name_without_L in names_without_L)==False:
    #         names_without_L.append(name_without_L)
    #     else:
    #         continue

    # only_seleceted_vars_df = df[names_without_L]


    column_names = list(df.loc[:, ~df.columns.isin(["Date"])])
    dfs_lags = []
    for name in column_names:
        new_columns = []
        for i in list(range(1, n+1)):
            new_columns.append(name + ".L"+str(i))

        df_lags = pd.DataFrame(columns=new_columns)
        for i in list(range(n)):
            df_lags[new_columns[i]] = df[name].shift(i+1)
        
        dfs_lags.append(df_lags)
    dfs_lags = pd.concat(dfs_lags, axis=1)

    if selected_f_names != None:
        dfs_lags = dfs_lags[selected_f_names]
    
    export_df = pd.concat([df[["Date", target]], dfs_lags], axis=1)
    export_df[target] = export_df[target].shift(1)
    target_lag1 = target+".L1"
    if target_lag1 in selected_f_names:
        export_df.pop(target+".L1")
    export_df["Date"] = export_df["Date"].shift(1)
    export_df = export_df.iloc[n:,:]
    export_df.reset_index(inplace=True, drop=True)

    return export_df


