import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


SMALL_SIZE = 36
MEDIUM_SIZE = 42
BIGGER_SIZE = 44

# Customising Matplotlib a little
plt.rc("patch", force_edgecolor = True)
plt.rc("grid", linewidth = 0.5)
plt.rc("axes", axisbelow = True)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fileCount = 0
directory = "Results_Room"

mean_rmse_overall_mean = []
mean_rmse_overall_std = []
mean_time_overall_mean = []
mean_time_overall_std = []
mean_outlier_overall_mean = []
mean_outlier_overall_std = []
mean_outlier_overall_class_mean = []
mean_outlier_overall_class_std = []

median_rmse_overall_mean = []
median_rmse_overall_std = []
median_time_overall_mean = []
median_time_overall_std = []
median_outlier_overall_mean = []
median_outlier_overall_std = []
median_outlier_overall_class_mean = []
median_outlier_overall_class_std = []

min_rmse_overall_mean = []
min_rmse_overall_std = []
min_time_overall_mean = []
min_time_overall_std = []
min_outlier_overall_mean = []
min_outlier_overall_std = []
min_outlier_overall_class_mean = []
min_outlier_overall_class_std = []

max_rmse_overall_mean = []
max_rmse_overall_std = []
max_time_overall_mean = []
max_time_overall_std = []
max_outlier_overall_mean = []
max_outlier_overall_std = []
max_outlier_overall_class_mean = []
max_outlier_overall_class_std = []

knn_rmse_overall_mean = []
knn_rmse_overall_std = []
knn_time_overall_mean = []
knn_time_overall_std = []
knn_outlier_overall_mean = []
knn_outlier_overall_std = []
knn_outlier_overall_class_mean = []
knn_outlier_overall_class_std = []

mice_rmse_overall_mean = []
mice_rmse_overall_std = []
mice_time_overall_mean = []
mice_time_overall_std = []
mice_outlier_overall_mean = []
mice_outlier_overall_std = []
mice_outlier_overall_class_mean = []
mice_outlier_overall_class_std = []

gain_rmse_overall_mean = []
gain_rmse_overall_std = []
gain_time_overall_mean = []
gain_time_overall_std = []
gain_outlier_overall_mean = []
gain_outlier_overall_std = []
gain_outlier_overall_class_mean = []
gain_outlier_overall_class_std = []

full_outlier_overall_mean = []
full_outlier_overall_std = []
full_outlier_overall_class_mean = []
full_outlier_overall_class_std = []



for folder in os.listdir(directory):
    temp_folder = directory + os.sep + folder

    knn_knn_rmse = []
    knn_knn_time = []
    knn_knn_outlier_reg = []
    knn_knn_outlier_class = []
    knn_rf_outlier_reg = []
    knn_rf_outlier_class = []
    
    mice_knn_rmse = []
    mice_knn_time = []
    mice_knn_outlier_reg = []
    mice_knn_outlier_class = []
    mice_rf_outlier_reg = []
    mice_rf_outlier_class = []
    
    gain_knn_rmse = []
    gain_knn_time = []
    gain_knn_outlier_reg = []
    gain_knn_outlier_class = []
    gain_rf_outlier_reg = []
    gain_rf_outlier_class = []
    
    mean_knn_rmse = []
    mean_knn_time = []
    mean_knn_outlier_reg = []
    mean_knn_outlier_class = []
    mean_rf_outlier_reg = []
    mean_rf_outlier_class = []
    
    median_knn_rmse = []
    median_knn_time = []
    median_knn_outlier_reg = []
    median_knn_outlier_class = []
    median_rf_outlier_reg = []
    median_rf_outlier_class = []
    
    min_knn_rmse = []
    min_knn_time = []
    min_knn_outlier_reg = []
    min_knn_outlier_class = []
    min_rf_outlier_reg = []
    min_rf_outlier_class = []
    
    max_knn_rmse = []
    max_knn_time = []
    max_knn_outlier_reg = []
    max_knn_outlier_class = []
    max_rf_outlier_reg = []
    max_rf_outlier_class = []
    
    full_knn_outlier_reg = []
    full_knn_outlier_class = []
    full_rf_outlier_reg = []
    full_rf_outlier_class = []

    for folder2 in os.listdir(temp_folder):
        # print(folder2)
        temp_folder2 = temp_folder + os.sep + folder2
        
        
        for file in os.listdir(temp_folder2):
            if "summary" in file:
                summ_file_name = temp_folder2 + os.sep + file
                summ_file = pd.read_csv(summ_file_name)
                
                knn_knn_rmse.append(summ_file["FillRMSE"][0])
                knn_knn_time.append(summ_file["FillTime"][0])
                knn_knn_outlier_reg.append(summ_file["OutlierRate"][0])
                knn_knn_outlier_class.append(summ_file["OutlierRate"][8])
                knn_rf_outlier_reg.append(summ_file["OutlierRate"][16])
                knn_rf_outlier_class.append(summ_file["OutlierRate"][24])
                
                mice_knn_rmse.append(summ_file["FillRMSE"][1])
                mice_knn_time.append(summ_file["FillTime"][1])
                mice_knn_outlier_reg.append(summ_file["OutlierRate"][1])
                mice_knn_outlier_class.append(summ_file["OutlierRate"][9])
                mice_rf_outlier_reg.append(summ_file["OutlierRate"][17])
                mice_rf_outlier_class.append(summ_file["OutlierRate"][25])
                
                gain_knn_rmse.append(summ_file["FillRMSE"][2])
                gain_knn_time.append(summ_file["FillTime"][2])
                gain_knn_outlier_reg.append(summ_file["OutlierRate"][2])
                gain_knn_outlier_class.append(summ_file["OutlierRate"][10])
                gain_rf_outlier_reg.append(summ_file["OutlierRate"][18])
                gain_rf_outlier_class.append(summ_file["OutlierRate"][26])
                
                mean_knn_rmse.append(summ_file["FillRMSE"][3])
                mean_knn_time.append(summ_file["FillTime"][3])
                mean_knn_outlier_reg.append(summ_file["OutlierRate"][3])
                mean_knn_outlier_class.append(summ_file["OutlierRate"][11])
                mean_rf_outlier_reg.append(summ_file["OutlierRate"][19])
                mean_rf_outlier_class.append(summ_file["OutlierRate"][27])
                
                median_knn_rmse.append(summ_file["FillRMSE"][4])
                median_knn_time.append(summ_file["FillTime"][4])
                median_knn_outlier_reg.append(summ_file["OutlierRate"][4])
                median_knn_outlier_class.append(summ_file["OutlierRate"][12])
                median_rf_outlier_reg.append(summ_file["OutlierRate"][20])
                median_rf_outlier_class.append(summ_file["OutlierRate"][28])
                
                min_knn_rmse.append(summ_file["FillRMSE"][5])
                min_knn_time.append(summ_file["FillTime"][5])
                min_knn_outlier_reg.append(summ_file["OutlierRate"][5])
                min_knn_outlier_class.append(summ_file["OutlierRate"][13])
                min_rf_outlier_reg.append(summ_file["OutlierRate"][21])
                min_rf_outlier_class.append(summ_file["OutlierRate"][29])
                
                max_knn_rmse.append(summ_file["FillRMSE"][6])
                max_knn_time.append(summ_file["FillTime"][6])
                max_knn_outlier_reg.append(summ_file["OutlierRate"][6])
                max_knn_outlier_class.append(summ_file["OutlierRate"][14])
                max_rf_outlier_reg.append(summ_file["OutlierRate"][22])
                max_rf_outlier_class.append(summ_file["OutlierRate"][30])
                
                full_knn_outlier_reg.append(summ_file["OutlierRate"][7])
                full_knn_outlier_class.append(summ_file["OutlierRate"][15])
                full_rf_outlier_reg.append(summ_file["OutlierRate"][23])
                full_rf_outlier_class.append(summ_file["OutlierRate"][31])


    knn_knn_rmse = [np.mean(knn_knn_rmse), np.std(knn_knn_rmse)]
    knn_knn_time = [np.mean(knn_knn_time), np.std(knn_knn_time)]
    knn_knn_outlier_reg = [np.mean(knn_knn_outlier_reg), np.std(knn_knn_outlier_reg)]
    knn_knn_outlier_class = [np.mean(knn_knn_outlier_class), np.std(knn_knn_outlier_class)]
    knn_rf_outlier_reg = [np.mean(knn_rf_outlier_reg), np.std(knn_rf_outlier_reg)]
    knn_rf_outlier_class = [np.mean(knn_rf_outlier_class), np.std(knn_rf_outlier_class)]
    
    mice_knn_rmse = [np.mean(mice_knn_rmse), np.std(mice_knn_rmse)]
    mice_knn_time = [np.mean(mice_knn_time), np.std(mice_knn_time)]
    mice_knn_outlier_reg = [np.mean(mice_knn_outlier_reg), np.std(mice_knn_outlier_reg)]
    mice_knn_outlier_class = [np.mean(mice_knn_outlier_class), np.std(mice_knn_outlier_class)]
    mice_rf_outlier_reg = [np.mean(mice_rf_outlier_reg), np.std(mice_rf_outlier_reg)]
    mice_rf_outlier_class = [np.mean(mice_rf_outlier_class), np.std(mice_rf_outlier_class)]
    
    gain_knn_rmse = [np.mean(gain_knn_rmse), np.std(gain_knn_rmse)]
    gain_knn_time = [np.mean(gain_knn_time), np.std(gain_knn_time)]
    gain_knn_outlier_reg = [np.mean(gain_knn_outlier_reg), np.std(gain_knn_outlier_reg)]
    gain_knn_outlier_class = [np.mean(gain_knn_outlier_class), np.std(gain_knn_outlier_class)]
    gain_rf_outlier_reg = [np.mean(gain_rf_outlier_reg), np.std(gain_rf_outlier_reg)]
    gain_rf_outlier_class = [np.mean(gain_rf_outlier_class), np.std(gain_rf_outlier_class)]
    
    mean_knn_rmse = [np.mean(mean_knn_rmse), np.std(mean_knn_rmse)]
    mean_knn_time = [np.mean(mean_knn_time), np.std(mean_knn_time)]
    mean_knn_outlier_reg = [np.mean(mean_knn_outlier_reg), np.std(mean_knn_outlier_reg)]
    mean_knn_outlier_class = [np.mean(mean_knn_outlier_class), np.std(mean_knn_outlier_class)]
    mean_rf_outlier_reg = [np.mean(mean_rf_outlier_reg), np.std(mean_rf_outlier_reg)]
    mean_rf_outlier_class = [np.mean(mean_rf_outlier_class), np.std(mean_rf_outlier_class)]
    
    median_knn_rmse = [np.mean(median_knn_rmse), np.std(median_knn_rmse)]
    median_knn_time = [np.mean(median_knn_time), np.std(median_knn_time)]
    median_knn_outlier_reg = [np.mean(median_knn_outlier_reg), np.std(median_knn_outlier_reg)]
    median_knn_outlier_class = [np.mean(median_knn_outlier_class), np.std(median_knn_outlier_class)]
    median_rf_outlier_reg = [np.mean(median_rf_outlier_reg), np.std(median_rf_outlier_reg)]
    median_rf_outlier_class = [np.mean(median_rf_outlier_class), np.std(median_rf_outlier_class)]
    
    min_knn_rmse = [np.mean(min_knn_rmse), np.std(min_knn_rmse)]
    min_knn_time = [np.mean(min_knn_time), np.std(min_knn_time)]
    min_knn_outlier_reg = [np.mean(min_knn_outlier_reg), np.std(min_knn_outlier_reg)]
    min_knn_outlier_class = [np.mean(min_knn_outlier_class), np.std(min_knn_outlier_class)]
    min_rf_outlier_reg = [np.mean(min_rf_outlier_reg), np.std(min_rf_outlier_reg)]
    min_rf_outlier_class = [np.mean(min_rf_outlier_class), np.std(min_rf_outlier_class)]
    
    max_knn_rmse = [np.mean(max_knn_rmse), np.std(max_knn_rmse)]
    max_knn_time = [np.mean(max_knn_time), np.std(max_knn_time)]
    max_knn_outlier_reg = [np.mean(max_knn_outlier_reg), np.std(max_knn_outlier_reg)]
    max_knn_outlier_class = [np.mean(max_knn_outlier_class), np.std(max_knn_outlier_class)]
    max_rf_outlier_reg = [np.mean(max_rf_outlier_reg), np.std(max_rf_outlier_reg)]
    max_rf_outlier_class = [np.mean(max_rf_outlier_class), np.std(max_rf_outlier_class)]
    
    full_knn_outlier_reg = [np.mean(full_knn_outlier_reg), np.std(full_knn_outlier_reg)]
    full_knn_outlier_class = [np.mean(full_knn_outlier_class), np.std(full_knn_outlier_class)]
    full_rf_outlier_reg = [np.mean(full_rf_outlier_reg), np.std(full_rf_outlier_reg)]
    full_rf_outlier_class = [np.mean(full_rf_outlier_class), np.std(full_rf_outlier_class)]
    print(folder, mice_knn_outlier_reg)


    error_lists = [     # Method, Time, RMSE, knnRegress, knnClass, rfRegress, rfClass
        ['Full', "-", "-",
        str(np.round(full_knn_outlier_reg[0], 2))     + " / " + str(np.round(full_knn_outlier_reg[1], 2)), 
        str(np.round(full_knn_outlier_class[0], 2))   + " / " + str(np.round(full_knn_outlier_class[1], 2)), 
        str(np.round(full_rf_outlier_reg[0], 2))      + " / " + str(np.round(full_rf_outlier_reg[1], 2)), 
        str(np.round(full_rf_outlier_class[0], 2))    + " / " + str(np.round(full_rf_outlier_class[1], 2))],
        ['Mean',  
        str(np.round(mean_knn_time[0], 2))            + " / " + str(np.round(mean_knn_time[1], 2)), 
        str(np.round(mean_knn_rmse[0], 2))            + " / " + str(np.round(mean_knn_rmse[1], 2)), 
        str(np.round(mean_knn_outlier_reg[0], 2))     + " / " + str(np.round(mean_knn_outlier_reg[1], 2)), 
        str(np.round(mean_knn_outlier_class[0], 2))     + " / " + str(np.round(mean_knn_outlier_class[1], 2)), 
        str(np.round(mean_rf_outlier_reg[0], 2))   + " / " + str(np.round(mean_rf_outlier_reg[1], 2)), 
        str(np.round(mean_rf_outlier_class[0], 2))      + " / " + str(np.round(mean_rf_outlier_class[1], 2))],
        ['Median',  
        str(np.round(median_knn_time[0], 2))          + " / " + str(np.round(median_knn_time[1], 2)), 
        str(np.round(median_knn_rmse[0], 2))          + " / " + str(np.round(median_knn_rmse[1], 2)), 
        str(np.round(median_knn_outlier_reg[0], 2))   + " / " + str(np.round(median_knn_outlier_reg[1], 2)), 
        str(np.round(median_knn_outlier_class[0], 2))   + " / " + str(np.round(median_knn_outlier_class[1], 2)), 
        str(np.round(median_rf_outlier_reg[0], 2)) + " / " + str(np.round(median_rf_outlier_reg[1], 2)), 
        str(np.round(median_rf_outlier_class[0], 2))    + " / " + str(np.round(median_rf_outlier_class[1], 2))],
        ['Minimum',  
        str(np.round(min_knn_time[0], 2))             + " / " + str(np.round(min_knn_time[1], 2)), 
        str(np.round(min_knn_rmse[0], 2))             + " / " + str(np.round(min_knn_rmse[1], 2)), 
        str(np.round(min_knn_outlier_reg[0], 2))      + " / " + str(np.round(min_knn_outlier_reg[1], 2)), 
        str(np.round(min_knn_outlier_class[0], 2))      + " / " + str(np.round(min_knn_outlier_class[1], 2)), 
        str(np.round(min_rf_outlier_reg[0], 2))    + " / " + str(np.round(min_rf_outlier_reg[1], 2)), 
        str(np.round(min_rf_outlier_class[0], 2))       + " / " + str(np.round(min_rf_outlier_class[1], 2))],
        ['Maximum',
        str(np.round(max_knn_time[0], 2))             + " / " + str(np.round(max_knn_time[1], 2)), 
        str(np.round(max_knn_rmse[0], 2))             + " / " + str(np.round(max_knn_rmse[1], 2)), 
        str(np.round(max_knn_outlier_reg[0], 2))      + " / " + str(np.round(max_knn_outlier_reg[1], 2)), 
        str(np.round(max_knn_outlier_class[0], 2))      + " / " + str(np.round(max_knn_outlier_class[1], 2)), 
        str(np.round(max_rf_outlier_reg[0], 2))    + " / " + str(np.round(max_rf_outlier_reg[1], 2)), 
        str(np.round(max_rf_outlier_class[0], 2))       + " / " + str(np.round(max_rf_outlier_class[1], 2))],
        ['kNN',      
        str(np.round(knn_knn_time[0], 2))             + " / " + str(np.round(knn_knn_time[1], 2)), 
        str(np.round(knn_knn_rmse[0], 2))             + " / " + str(np.round(knn_knn_rmse[1], 2)), 
        str(np.round(knn_knn_outlier_reg[0], 2))      + " / " + str(np.round(knn_knn_outlier_reg[1], 2)), 
        str(np.round(knn_knn_outlier_class[0], 2))      + " / " + str(np.round(knn_knn_outlier_class[1], 2)), 
        str(np.round(knn_rf_outlier_reg[0], 2))    + " / " + str(np.round(knn_rf_outlier_reg[1], 2)), 
        str(np.round(knn_rf_outlier_class[0], 2))       + " / " + str(np.round(knn_rf_outlier_class[1], 2))],
        ['MICE',      
        str(np.round(mice_knn_time[0], 2))            + " / " + str(np.round(mice_knn_time[1], 2)), 
        str(np.round(mice_knn_rmse[0], 2))            + " / " + str(np.round(mice_knn_rmse[1], 2)), 
        str(np.round(mice_knn_outlier_reg[0], 2))     + " / " + str(np.round(mice_knn_outlier_reg[1], 2)), 
        str(np.round(mice_knn_outlier_class[0], 2))     + " / " + str(np.round(mice_knn_outlier_class[1], 2)), 
        str(np.round(mice_rf_outlier_reg[0], 2))   + " / " + str(np.round(mice_rf_outlier_reg[1], 2)), 
        str(np.round(mice_rf_outlier_class[0], 2))      + " / " + str(np.round(mice_rf_outlier_class[1], 2))],
        ['GAIN',    
        str(np.round(gain_knn_time[0], 2))            + " / " + str(np.round(gain_knn_time[1], 2)), 
        str(np.round(gain_knn_rmse[0], 2))            + " / " + str(np.round(gain_knn_rmse[1], 2)), 
        str(np.round(gain_knn_outlier_reg[0], 2))     + " / " + str(np.round(gain_knn_outlier_reg[1], 2)), 
        str(np.round(gain_knn_outlier_class[0], 2))     + " / " + str(np.round(gain_knn_outlier_class[1], 2)), 
        str(np.round(gain_rf_outlier_reg[0], 2))   + " / " + str(np.round(gain_rf_outlier_reg[1], 2)), 
        str(np.round(gain_rf_outlier_class[0], 2))      + " / " + str(np.round(gain_rf_outlier_class[1], 2))],
        ]

    error_frame = pd.DataFrame(error_lists, columns=["Imputation Method", "Imputation Time", "Imputation RMSE", "kNN Regress Outlier Rate",  
        "kNN Class Outlier Rate", "RF Regress Outlier Rate", "RF Class Outlier Rate"])
    error_frame.to_csv(folder + ".csv", index=False)
    error_frame.to_latex(folder + ".tex", index=False, float_format="%.2f", caption=folder, column_format='lcccccc')

    mean_rmse_overall_mean      .append(mean_knn_rmse[0]       )
    mean_rmse_overall_std       .append(mean_knn_rmse[1]       )
    mean_time_overall_mean      .append(mean_knn_time[0]       )
    mean_time_overall_std       .append(mean_knn_time[1]       )
    mean_outlier_overall_mean   .append(mean_knn_outlier_reg[0])
    mean_outlier_overall_std    .append(mean_knn_outlier_reg[1])
    mean_outlier_overall_class_mean   .append(mean_knn_outlier_class[0])
    mean_outlier_overall_class_std    .append(mean_knn_outlier_class[1])

    median_rmse_overall_mean    .append(median_knn_rmse[0]       )
    median_rmse_overall_std     .append(median_knn_rmse[1]       )
    median_time_overall_mean    .append(median_knn_time[0]       )
    median_time_overall_std     .append(median_knn_time[1]       )
    median_outlier_overall_mean .append(median_knn_outlier_reg[0])
    median_outlier_overall_std  .append(median_knn_outlier_reg[1])
    median_outlier_overall_class_mean .append(median_knn_outlier_class[0])
    median_outlier_overall_class_std  .append(median_knn_outlier_class[1])

    min_rmse_overall_mean       .append(min_knn_rmse[0]       )
    min_rmse_overall_std        .append(min_knn_rmse[1]       )
    min_time_overall_mean       .append(min_knn_time[0]       )
    min_time_overall_std        .append(min_knn_time[1]       )
    min_outlier_overall_mean    .append(min_knn_outlier_reg[0])
    min_outlier_overall_std     .append(min_knn_outlier_reg[1])
    min_outlier_overall_class_mean    .append(min_knn_outlier_class[0])
    min_outlier_overall_class_std     .append(min_knn_outlier_class[1])

    max_rmse_overall_mean       .append(max_knn_rmse[0]       )
    max_rmse_overall_std        .append(max_knn_rmse[1]       )
    max_time_overall_mean       .append(max_knn_time[0]       )
    max_time_overall_std        .append(max_knn_time[1]       )
    max_outlier_overall_mean    .append(max_knn_outlier_reg[0])
    max_outlier_overall_std     .append(max_knn_outlier_reg[1])
    max_outlier_overall_class_mean    .append(max_knn_outlier_class[0])
    max_outlier_overall_class_std     .append(max_knn_outlier_class[1])

    knn_rmse_overall_mean       .append(knn_knn_rmse[0]       )
    knn_rmse_overall_std        .append(knn_knn_rmse[1]       )
    knn_time_overall_mean       .append(knn_knn_time[0]       )
    knn_time_overall_std        .append(knn_knn_time[1]       )
    knn_outlier_overall_mean    .append(knn_knn_outlier_reg[0])
    knn_outlier_overall_std     .append(knn_knn_outlier_reg[1])
    knn_outlier_overall_class_mean    .append(knn_knn_outlier_class[0])
    knn_outlier_overall_class_std     .append(knn_knn_outlier_class[1])

    mice_rmse_overall_mean      .append(mice_knn_rmse[0]       )
    mice_rmse_overall_std       .append(mice_knn_rmse[1]       )
    mice_time_overall_mean      .append(mice_knn_time[0]       )
    mice_time_overall_std       .append(mice_knn_time[1]       )
    mice_outlier_overall_mean   .append(mice_knn_outlier_reg[0])
    mice_outlier_overall_std    .append(mice_knn_outlier_reg[1])
    mice_outlier_overall_class_mean   .append(mice_knn_outlier_class[0])
    mice_outlier_overall_class_std    .append(mice_knn_outlier_class[1])

    gain_rmse_overall_mean      .append(gain_knn_rmse[0]       )
    gain_rmse_overall_std       .append(gain_knn_rmse[1]       )
    gain_time_overall_mean      .append(gain_knn_time[0]       )
    gain_time_overall_std       .append(gain_knn_time[1]       )
    gain_outlier_overall_mean   .append(gain_knn_outlier_reg[0])
    gain_outlier_overall_std    .append(gain_knn_outlier_reg[1])
    gain_outlier_overall_class_mean   .append(gain_knn_outlier_class[0])
    gain_outlier_overall_class_std    .append(gain_knn_outlier_class[1])
    
    full_outlier_overall_mean   .append(full_knn_outlier_reg[0])
    full_outlier_overall_std    .append(full_knn_outlier_reg[1])
    full_outlier_overall_class_mean   .append(full_knn_outlier_class[0])
    full_outlier_overall_class_std    .append(full_knn_outlier_class[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(35,11))#, gridspec_kw = {'wspace':0, 'hspace':0})
x_ticklabels = ["2", "5", "10", "15", "20", "25", "30"]
lw_arg = 3
ms_arg = 20
alpha_arg = 1
x_tick_vals = range(len(x_ticklabels))

ax1.errorbar(x_tick_vals, mean_rmse_overall_mean, mean_rmse_overall_std, label="Mean", linewidth=3, marker="o", markersize=ms_arg, color="tab:blue", alpha=alpha_arg, linestyle="dotted")
ax1.errorbar(x_tick_vals, median_rmse_overall_mean, median_rmse_overall_std, label="Median", linewidth=3, marker="o", markersize=ms_arg, color="tab:orange", alpha=alpha_arg, linestyle="dashed")
ax1.errorbar(x_tick_vals, min_rmse_overall_mean, min_rmse_overall_std, label="Minimum", linewidth=3, marker="o", markersize=ms_arg, color="tab:green", alpha=alpha_arg, linestyle="dashdot")
ax1.errorbar(x_tick_vals, max_rmse_overall_mean, max_rmse_overall_std, label="Maximum", linewidth=3, marker="o", markersize=ms_arg, color="tab:red", alpha=alpha_arg, linestyle=(0, (5, 10)))
ax1.errorbar(x_tick_vals, knn_rmse_overall_mean, knn_rmse_overall_std, label="kNN", linewidth=3, marker="o", markersize=ms_arg, color="tab:purple", alpha=alpha_arg, linestyle=(0, (5, 1)))
ax1.errorbar(x_tick_vals, mice_rmse_overall_mean, mice_rmse_overall_std, label="MICE", linewidth=3, marker="o", markersize=ms_arg, color="tab:brown", alpha=alpha_arg, linestyle=(0, (1, 1)))
ax1.errorbar(x_tick_vals, gain_rmse_overall_mean, gain_rmse_overall_std, label="GAIN", linewidth=3, marker="o", markersize=ms_arg, color="tab:pink", alpha=alpha_arg, linestyle=(0, (3, 10, 1, 10)))
ax1.text(-1.25, 0.8, "A", fontsize=56, fontweight="bold")
ax1.set_xticks(x_tick_vals)
ax1.grid(axis='y', which="major", ls="-")
ax1.grid(axis='y', which="minor", ls=":")
ax1.set_xticklabels(x_ticklabels)
ax1.set_ylabel("RMSE")
ax1.set_xlabel("Percentage of missing data")
ax1.set_yscale("log")
alpha_arg = 0.8
ms_arg = 20

ax2.set_ylabel("Outlier Rate (%)")
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.errorbar(x_tick_vals, mean_outlier_overall_mean, mean_outlier_overall_std, label="Mean", linewidth=lw_arg, marker="v", markersize=ms_arg, color="tab:blue", alpha=alpha_arg, linestyle="dotted")
ax2.errorbar(x_tick_vals, median_outlier_overall_mean, median_outlier_overall_std, label="Median", linewidth=lw_arg, marker="^", markersize=ms_arg, color="tab:orange", alpha=alpha_arg, linestyle="dashed")
ax2.errorbar(x_tick_vals, min_outlier_overall_mean, min_outlier_overall_std, label="Minimum", linewidth=lw_arg, marker="s", markersize=ms_arg, color="tab:green", alpha=alpha_arg, linestyle="dashdot")
ax2.errorbar(x_tick_vals, max_outlier_overall_mean, max_outlier_overall_std, label="Maximum", linewidth=lw_arg, marker="p", markersize=ms_arg, color="tab:red", alpha=alpha_arg, linestyle=(0, (5, 10)))#"loosely dashed")
ax2.errorbar(x_tick_vals, knn_outlier_overall_mean, knn_outlier_overall_std, label="kNN", linewidth=lw_arg, marker="P", markersize=ms_arg, color="tab:purple", alpha=alpha_arg, linestyle=(0, (5, 1)))#"densely dashed")
ax2.errorbar(x_tick_vals, mice_outlier_overall_mean, mice_outlier_overall_std, label="MICE", linewidth=lw_arg, marker="X", markersize=ms_arg, color="tab:brown", alpha=alpha_arg, linestyle=(0, (1, 1)))#"densely dotted")
ax2.errorbar(x_tick_vals, gain_outlier_overall_mean, gain_outlier_overall_std, label="GAIN", linewidth=lw_arg, marker="D", markersize=ms_arg, color="tab:pink", alpha=alpha_arg, linestyle=(0, (3, 10, 1, 10)))#"loosely dashdotted")
ax2.errorbar(x_tick_vals, full_outlier_overall_mean, full_outlier_overall_std, label="Baseline", linewidth=lw_arg, marker="o", markersize=ms_arg, color="tab:grey", alpha=alpha_arg, linestyle= (0, (3, 5, 1, 5, 1, 5)))#"dashdotdotted")

ax2.errorbar(x_tick_vals, mean_outlier_overall_class_mean, mean_outlier_overall_class_std, label="Mean", linewidth=lw_arg, marker="v", markersize=ms_arg, fillstyle="none", color="tab:blue", alpha=alpha_arg, linestyle="dotted")
ax2.errorbar(x_tick_vals, median_outlier_overall_class_mean, median_outlier_overall_class_std, label="Median", linewidth=lw_arg, marker="^", markersize=ms_arg, fillstyle="none", color="tab:orange", alpha=alpha_arg, linestyle="dashed")
ax2.errorbar(x_tick_vals, min_outlier_overall_class_mean, min_outlier_overall_class_std, label="Minimum", linewidth=lw_arg, marker="s", markersize=ms_arg, fillstyle="none", color="tab:green", alpha=alpha_arg, linestyle="dashdot")
ax2.errorbar(x_tick_vals, max_outlier_overall_class_mean, max_outlier_overall_class_std, label="Maximum", linewidth=lw_arg, marker="p", markersize=ms_arg, fillstyle="none", color="tab:red", alpha=alpha_arg, linestyle=(0, (5, 10)))#"loosely dashed")
ax2.errorbar(x_tick_vals, knn_outlier_overall_class_mean, knn_outlier_overall_class_std, label="kNN", linewidth=lw_arg, marker="P", markersize=ms_arg, fillstyle="none", color="tab:purple", alpha=alpha_arg, linestyle=(0, (5, 1)))#"densely dashed")
ax2.errorbar(x_tick_vals, mice_outlier_overall_class_mean, mice_outlier_overall_class_std, label="MICE", linewidth=lw_arg, marker="X", markersize=ms_arg, fillstyle="none", color="tab:brown", alpha=alpha_arg, linestyle=(0, (1, 1)))#"densely dotted")
ax2.errorbar(x_tick_vals, gain_outlier_overall_class_mean, gain_outlier_overall_class_std, label="GAIN", linewidth=lw_arg, marker="D", markersize=ms_arg, fillstyle="none", color="tab:pink", alpha=alpha_arg, linestyle=(0, (3, 10, 1, 10)))#"loosely dashdotted")
ax2.errorbar(x_tick_vals, full_outlier_overall_class_mean, full_outlier_overall_class_std, label="Baseline", linewidth=lw_arg, marker="o", markersize=ms_arg, fillstyle="none", color="tab:grey", alpha=alpha_arg, linestyle= (0, (3, 5, 1, 5, 1, 5)))#"dashdotdotted")

ax2.text(-0.7, 73, "B", fontsize=56, fontweight="bold")

ax2.set_xticks(x_tick_vals)
ax2.set_xticklabels(x_ticklabels)
ax2.grid(axis='y')
ax2.set_xlabel("Percentage of missing data")

mean_legend = plt.Line2D([0], [0], color="tab:blue", lw=lw_arg, marker="v", markersize=ms_arg, label='Mean', linestyle="dotted")
median_legend = plt.Line2D([0], [0], color="tab:orange", lw=lw_arg, marker="^", markersize=ms_arg, label='Median', linestyle="dashed")
min_legend = plt.Line2D([0], [0], color="tab:green", lw=lw_arg, marker="s", markersize=ms_arg, label='Minimum', linestyle="dashdot")
max_legend = plt.Line2D([0], [0], color="tab:red", lw=lw_arg, marker="p", markersize=ms_arg, label='Maximum', linestyle=(0, (5, 10)))#"loosely dashed")
knn_legend = plt.Line2D([0], [0], color="tab:purple", lw=lw_arg, marker="P", markersize=ms_arg, label='kNN', linestyle=(0, (5, 1)))#"densely dashed")
mice_legend = plt.Line2D([0], [0], color="tab:brown", lw=lw_arg, marker="X", markersize=ms_arg, label='MICE', linestyle=(0, (1, 1)))#"densely dotted")
gain_legend = plt.Line2D([0], [0], color="tab:pink", lw=lw_arg, marker="D", markersize=ms_arg, label='GAIN', linestyle=(0, (3, 10, 1, 10)))#"loosely dashdotted")
full_legend = plt.Line2D([0], [0], color="tab:grey", lw=lw_arg, marker="o", markersize=ms_arg, label='Full', linestyle= (0, (3, 5, 1, 5, 1, 5)))#"dashdotdotted")
regress_legend = plt.Line2D([0], [0], color="white", mec = 'black', mfc = 'black', lw=lw_arg, label='Regression', marker="o", markersize=ms_arg)
class_legend = plt.Line2D([0], [0], color="white", mec = 'black', lw=lw_arg, label='Classification', marker="o", fillstyle="none", markersize=ms_arg)


leg = fig.legend(handles=[mean_legend, median_legend, min_legend, max_legend, knn_legend, mice_legend, gain_legend, full_legend, 
    regress_legend, class_legend], loc='lower center', ncol=5, bbox_to_anchor=(0.52, -0.17))#, bbox_to_anchor=(1, 0.5)), frameon=False, framealpha=0

# fig.subplots_adjust(bottom=0.75)

# legend_elements = [plt.Line2D([0], [0], color='b', lw=4, label='Line'),
#                    plt.Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='g', markersize=15),
#                    Patch(facecolor='orange', edgecolor='r',
#                          label='Color Patch')]

# # Create the figure
# ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))


# plt.tight_layout(pad=0.5)
plt.savefig("Outlier_Rates.pdf", bbox_extra_artists=(leg,), bbox_inches='tight')


# fig, ax1 = plt.subplots(figsize=(16,9))
# x_ticklabels = ["0", "2", "5", "10", "15", "20", "25", "30"]
# x_tick_vals = range(len(x_ticklabels)-1)
# ax1.set_ylabel("RMSE")
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax1.grid()
# # ax1.set_yscale("log")
# ax1.set_xticklabels(x_ticklabels)
# # ax1.set_title("RMSE of each Imputed Dataset")
# plt.tight_layout()
# plt.savefig("RMSE.pdf")
# print(gain_time_overall_std)
