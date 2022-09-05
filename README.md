# MSIN0114

The code for MSIN0114 results replication.

df_apple - Apple data

hungury_chickenpox - Chickenpox data

MAIN_STAGE_ONE - Stage 1 algorithm code

MAIN_STAGE_TWO - Stage 2 algorithm code

Classical_FS - Defines functions for conventional feature serlection (FS) techniques

Sun_Model_Class - Defines class to store stage 1 results

my_metrics - defines functions for varipus error measures

lag_functions - defines functions to create lags of the featurest in a necessery formatting

STAGE_TWO_REVERSE - Function to replicate dropping most important features first

MAIN_APPLE_1 - Run stage 1 on apple data and store the results

MAIN_APPLE_2 - Run stage 2 on apple data and store the results

Other_FS_Apple - Run conventional feature selectoion (FS) methods on original Apple data

Other_FS_stage_two_Apple - Run for conventional to select as many features as the stage 2 in Apple data

apple_results_vis - replicate stage 1 MAE graph

apple_results_vis_stage_2 - replicate stage 2 MAE graph

apple_results_vis_stage_2_vs_evoML - replicate the comparison between our method and evoL recommended

MAIN_CHICKENPOX_1 - Run stage 1 on chickenpox data and store the results

MAIN_CHICKENPOX_1 - Run stage 1 on CHICKENPOX data and store the results

MAIN_CHICKENPOX_2 - Run stage 2 on CHICKENPOX data and store the results

Other_FS_CHICKENPOX - Run conventional feature selectoion (FS) methods on original CHICKENPOX data

Other_FS_stage_two_CHICKENPOX - Run for conventional to select as many features as the stage 2 in CHICKENPOX data

Each DATASET (Apple/Chokenpox) Folder contain:

Data - subsets selected to be run on evoML (special formatting) and elswhere

evoML_models_hyperparameters - hyperparameters returned by the evoML platform for each tuned model

Figures - figures related to the dataset

Tables - tables related to the dataset
