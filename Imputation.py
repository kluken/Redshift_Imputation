import argparse, os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import KFold 
from redshift_utils import *
from gain_utils import blank_data
from gain_utils import rmse_loss
from MICE import *
from gain import gain
from knnImputer import kNN_impute



# How to blank data:
#   - import the blank_data function (from gain_utils import blank_data)
#   - use the blank_data method, giving it the 2-d array you want to blank (typically the x_vals), the fraction you want blanked, and a random generator to use (np.random.default_rng(rand_seed))


#When using MICE:
#   - Instantiate a new object, giving it the maxiters (and if you don't want the TQDM bar, tqdm_disable = True)
#   - Use newMiceObj.benchmark(Original_Dataframe, Dataframe_with_NaNs) which will return a list of dictionarys that has the loss results as the first return val, and the dataframe with imputed values as the second. 
#   - OR! If you don't want the MICE errors calculated, you can just get the imputer dataframe with the newMicrObj.fill_missing_values(Datafram_with_NaNs) function. Can use the gain rmse method for this too. 

#When using GAIN:
#   - Import gain (from gain import gain)
#   - Use gain (gain(miss_data_x, gain_parameters)). Default gain parameters used by Rabina were: batch size = 128, hint rate = 0.9, alpha = 100, iterations = 60000. Returns the imputed data. 
#   - To calculate rmse, use the rmse_loss method (from gain_utils import rmse_loss), giving it the original data, the imputed data, and the missing mask. 


def main():
    starting_time = datetime.now()
    # Sizes for plots
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

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

    # Program Defaults:
    parser = argparse.ArgumentParser(description="This script does stuff.")
    parser.add_argument("-s", "--seed", nargs=1, required=True, type=int, help="Random Seed to use") 
    parser.add_argument("-m", "--missingRate", nargs=1, required=True, type=float, help="Decimal value used to set the missing percentage") 
    parser.add_argument("-t", "--tqdmDisable", action='store_true', help="Disable the TQDM bar") 

    args = vars(parser.parse_args())


    initial_rand_seed = args["seed"][0]  # argument
    initial_rand_gen = np.random.default_rng(initial_rand_seed) # Initial Random Generator to use
    split_data_seed = initial_rand_gen.integers(314159265) # Generate random seed to use to split data
    missing_data_seed = initial_rand_gen.integers(314159265) # Generate random seed to use to determine missing data
    mice_seed = initial_rand_gen.integers(314159265) # Generate random seed to use within MICE
    gain_seed = initial_rand_gen.integers(314159265) # Generate random seed to use within GAIN
    knn_seed = initial_rand_gen.integers(314159265) # Generate random seed to use for kNN cross-validation
    tree_seed = initial_rand_gen.integers(314159265) # Generate random seed to use for Random Forest
    missing_percentage = args["missingRate"][0] # Percentage of data to be set to missing, generally a value [2 5 10 15 20 25 30]
    tqdm_disable_bool = args["tqdmDisable"] # Disable TQDM progress bars if running many
    knn_distance = 99 # Mahalanobis. Can be swapped later if needed, but will probably stay. 1 for Manhattan, 2 for Euclidean, 99 for Mahalanobis
    num_class_bins = 15 # Number of bins to use for classification if needed
    kfold_splits = 10 # Used in k-Fold Cross Validation.
    catalogue = "ATLAS_Complete_fixed.fits" # Dataset to use.
    data_cols = ["z","Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"] # Columns to use.
    k_range_reg = np.arange(3, 23)
    k_range_class = np.arange(3, 43, 2)
    tree_range = np.arange(1, 61, 1)


    full_dataset = read_fits(catalogue, data_cols)
    x_vals = full_dataset[:, 1:]
    y_vals = full_dataset[:,0]

    x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(x_vals, y_vals, np.random.default_rng(split_data_seed))

    y_vals_class, bin_edges, bin_median = bin_data_func(np.copy(y_vals), num_class_bins)
    _, _, y_vals_class_train, y_vals_class_test = split_data(x_vals, y_vals_class, np.random.default_rng(split_data_seed))

    x_vals_blank, missing_mask = blank_data(x_vals_test, missing_percentage, np.random.default_rng(missing_data_seed))

    x_vals_knn = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_knn, knn_impute_k = kNN_impute(x_vals_knn, x_vals_train, np.random.default_rng(knn_seed), missing_percentage, tqdm_disable=tqdm_disable_bool)
    knn_time = (datetime.now() - start_time)
    knn_rmse = rmse_loss(x_vals_test, x_vals_knn, missing_mask)

    start_time = datetime.now()
    mice_obj = FastMICE(rand_generator=np.random.default_rng(mice_seed), tqdm_disable=tqdm_disable_bool)
    x_vals_mice = pd.DataFrame(x_vals_blank)
    x_vals_mice = np.array(mice_obj.fill_missing_values(x_vals_mice))
    mice_time = (datetime.now() - start_time)
    mice_rmse = rmse_loss(x_vals_test, x_vals_mice, missing_mask)

    gain_parset = {'batch_size': 128,
                    'hint_rate': 0.9,
                    'alpha': 100,
                    'iterations': 60000}
    start_time = datetime.now()
    
    x_vals_gain = gain(x_vals_blank, gain_parset, np.random.default_rng(gain_seed), tqdm_disable=tqdm_disable_bool)
    gain_time = (datetime.now() - start_time)
    gain_rmse = rmse_loss(x_vals_test, x_vals_gain, missing_mask)
    

    x_vals_mean = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_mean = simple_impute(x_vals_mean, np.nanmean, np.nan)
    mean_time = (datetime.now() - start_time)
    mean_rmse = rmse_loss(x_vals_test, x_vals_mean, missing_mask)


    x_vals_min = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_min = simple_impute(x_vals_min, np.nanmin, np.nan)
    min_time = (datetime.now() - start_time)
    min_rmse = rmse_loss(x_vals_test, x_vals_min, missing_mask)

    x_vals_max = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_max = simple_impute(x_vals_max, np.nanmax, np.nan)
    max_time = (datetime.now() - start_time)
    max_rmse = rmse_loss(x_vals_test, x_vals_max, missing_mask)

    x_vals_median = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_median = simple_impute(x_vals_median, np.nanmedian, np.nan)
    median_time = (datetime.now() - start_time)
    median_rmse = rmse_loss(x_vals_test, x_vals_median, missing_mask)



    # Collection Arrays
    outlier_knn_regress = []
    outlier_knn_class = []
    outlier_random_regress = []
    outlier_random_class = []
    mse_knn_regress = []
    acc_knn_class = []
    mse_random_regress = []
    acc_random_class = []

    for i in tqdm(range(len(k_range_reg)), disable=tqdm_disable_bool):
        #setup arrays
        outrate_cross_regress = []
        mse_cross_regress = []
        outrate_cross_class = []
        accuracy_cross_class = []
        kFold = KFold(n_splits=kfold_splits, random_state=split_data_seed, shuffle=True)
        for train_index, test_index in kFold.split(x_vals_train):
            #Set up datasets - x vals will be shared between regression and classification
            x_vals_cross_train = x_vals_train[train_index]
            x_vals_cross_test = x_vals_train[test_index]
            # y vals for regression tests
            y_vals_cross_train = y_vals_train[train_index]
            y_vals_cross_test = y_vals_train[test_index]
            # y vals for classification tests
            y_vals_cross_train_class = y_vals_class_train[train_index]
            y_vals_cross_test_class = y_vals_class_train[test_index]
            
            #Normalise x-values
            x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_cross_train, x_vals_cross_test)

            #predict y_vals for regression
            pred_reg, mse_reg = kNN(k_range_reg[i], x_vals_train_norm, x_vals_test_norm, y_vals_cross_train, y_vals_cross_test, knn_distance)
            
            #predict y_vals for classification
            pred_class, acc_class = kNN_classification(k_range_class[i], x_vals_train_norm, x_vals_test_norm, y_vals_cross_train_class, y_vals_cross_test_class, knn_distance)

            #calculate outlier_rate
            reg_outlier = outlier_rate(norm_residual(y_vals_cross_test, pred_reg))
            class_outlier = outlier_rate(norm_residual(y_vals_cross_test_class, pred_class))

            # Save outlier rates
            outrate_cross_class.append(class_outlier)
            accuracy_cross_class.append(acc_class)
            outrate_cross_regress.append(reg_outlier)
            mse_cross_regress.append(mse_reg)
            
        # Accumulate the average errors from the cross-validation slices
        outlier_knn_regress.append(np.mean(outrate_cross_regress))
        outlier_knn_class.append(np.mean(outrate_cross_class))
        mse_knn_regress.append(np.mean(mse_cross_regress))
        acc_knn_class.append(np.mean(accuracy_cross_class))

    tree_rand_gen = np.random.default_rng(tree_seed)
    for tree in tqdm(tree_range, disable=tqdm_disable_bool):
        #setup arrays
        outrate_cross_regress = []
        mse_cross_regress = []
        outrate_cross_class = []
        accuracy_cross_class = []
        kFold = KFold(n_splits=kfold_splits, random_state=split_data_seed, shuffle=True)
        for train_index, test_index in kFold.split(x_vals_train):
            #Set up datasets - x vals will be shared between regression and classification
            x_vals_cross_train = x_vals_train[train_index]
            x_vals_cross_test = x_vals_train[test_index]
            # y vals for regression tests
            y_vals_cross_train = y_vals_train[train_index]
            y_vals_cross_test = y_vals_train[test_index]
            # y vals for classification tests
            y_vals_cross_train_class = y_vals_class_train[train_index]
            y_vals_cross_test_class = y_vals_class_train[test_index]
            
            #Normalise x-values
            x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_cross_train, x_vals_cross_test)

            # Get new random seed for random forest

            rf_seed = tree_rand_gen.integers(314159265)

            #predict y_vals for regression
            pred_reg, mse_reg = random_forest_regress(tree, x_vals_train_norm, x_vals_test_norm, y_vals_cross_train, y_vals_cross_test, rf_seed)
            #predict y_vals for classification
            pred_class, acc_class = random_forest_regress(tree, x_vals_train_norm, x_vals_test_norm, y_vals_cross_train_class, y_vals_cross_test_class, rf_seed)

            #calculate outlier_rate
            reg_outlier = outlier_rate(norm_residual(y_vals_cross_test, pred_reg))
            class_outlier = outlier_rate(norm_residual(y_vals_cross_test_class, pred_class))

            # Save outlier rates
            outrate_cross_class.append(class_outlier)
            accuracy_cross_class.append(acc_class)
            outrate_cross_regress.append(reg_outlier)
            mse_cross_regress.append(mse_reg)
            
        # Accumulate the average errors from the cross-validation slices
        outlier_random_regress.append(np.mean(outrate_cross_regress))
        outlier_random_class.append(np.mean(outrate_cross_class))
        mse_random_regress.append(np.mean(mse_cross_regress))
        acc_random_class.append(np.mean(accuracy_cross_class))

    best_k_reg = k_range_reg[np.argmin(outlier_knn_regress)]
    best_k_class = k_range_class[np.argmin(outlier_knn_class)]
    best_tree_reg = tree_range[np.argmin(outlier_random_regress)]
    best_tree_class =  tree_range[np.argmin(outlier_random_class)]

    # Normalise x_vals

    # knn regression
    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_knn)
    pred_reg_knn_knn, mse_reg_knn_knn = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_knn, mse_class_knn_knn = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_knn, mse_reg_rf_knn = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_knn, mse_class_rf_knn = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_mice)
    pred_reg_knn_mice, mse_reg_knn_mice = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_mice, mse_class_knn_mice = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_mice, mse_reg_rf_mice = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_mice, mse_class_rf_mice = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_gain)
    pred_reg_knn_gain, mse_reg_knn_gain = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_gain, mse_class_knn_gain = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_gain, mse_reg_rf_gain = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_gain, mse_class_rf_gain = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_median)
    pred_reg_knn_median, mse_reg_knn_median = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_median, mse_class_knn_median = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_median, mse_reg_rf_median = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_median, mse_class_rf_median = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_mean)
    pred_reg_knn_mean, mse_reg_knn_mean = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_mean, mse_class_knn_mean = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_mean, mse_reg_rf_mean = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_mean, mse_class_rf_mean = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_min)
    pred_reg_knn_min, mse_reg_knn_min = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_min, mse_class_knn_min = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_min, mse_reg_rf_min = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_min, mse_class_rf_min = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_max)
    pred_reg_knn_max, mse_reg_knn_max = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_max, mse_class_knn_max = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_max, mse_reg_rf_max = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_max, mse_class_rf_max = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)
    
    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_test)
    pred_reg_knn_full, mse_reg_knn_full = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_full, mse_class_knn_full = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, knn_distance)
    pred_reg_rf_full, mse_reg_rf_full = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_full, mse_class_rf_full = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_class_train, y_vals_class_test, rf_seed)

    print(datetime.now() - starting_time)

    # Calculate outlier rates
    outlier_reg_knn_knn = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_knn))
    outlier_class_knn_knn = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_knn))
    outlier_reg_rf_knn = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_knn))
    outlier_class_rf_knn = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_knn))

    outlier_reg_knn_mice = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_mice))
    outlier_class_knn_mice = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_mice))
    outlier_reg_rf_mice = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_mice))
    outlier_class_rf_mice = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_mice))

    outlier_reg_knn_gain = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_gain))
    outlier_class_knn_gain = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_gain))
    outlier_reg_rf_gain = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_gain))
    outlier_class_rf_gain = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_gain))

    outlier_reg_knn_mean = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_mean))
    outlier_class_knn_mean = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_mean))
    outlier_reg_rf_mean = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_mean))
    outlier_class_rf_mean = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_mean))

    outlier_reg_knn_median = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_median))
    outlier_class_knn_median = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_median))
    outlier_reg_rf_median = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_median))
    outlier_class_rf_median = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_median))

    outlier_reg_knn_min = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_min))
    outlier_class_knn_min = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_min))
    outlier_reg_rf_min = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_min))
    outlier_class_rf_min = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_min))

    outlier_reg_knn_max = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_max))
    outlier_class_knn_max = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_max))
    outlier_reg_rf_max = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_max))
    outlier_class_rf_max = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_max))

    outlier_reg_knn_full = outlier_rate(norm_residual(y_vals_test, pred_reg_knn_full))
    outlier_class_knn_full = outlier_rate(norm_residual(y_vals_class_test, pred_class_knn_full))
    outlier_reg_rf_full = outlier_rate(norm_residual(y_vals_test, pred_reg_rf_full))
    outlier_class_rf_full = outlier_rate(norm_residual(y_vals_class_test, pred_class_rf_full))

    # Compile error data frame
    # Fill Method, FillKVal, FillTime, FillRMSE, Prediction Method, k/tree_val, Class/Regress, Out_Rate, MSE/Acc
    error_lists = [['kNN', knn_impute_k, str(knn_time.total_seconds()), knn_rmse, "kNN", best_k_reg, "Regression", outlier_reg_knn_knn, mse_reg_knn_knn], 
        ['MICE', "", str(mice_time.total_seconds()), mice_rmse, "kNN", best_k_reg, "Regression", outlier_reg_knn_mice, mse_reg_knn_mice],  
        ['GAIN', "", str(gain_time.total_seconds()), gain_rmse, "kNN", best_k_reg, "Regression", outlier_reg_knn_gain, mse_reg_knn_gain],  
        ['Mean', "", str(mean_time.total_seconds()), mean_rmse, "kNN", best_k_reg, "Regression", outlier_reg_knn_mean, mse_reg_knn_mean],  
        ['Median', "", str(median_time.total_seconds()), median_rmse, "kNN", best_k_reg, "Regression", outlier_reg_knn_median, mse_reg_knn_median],  
        ['Min', "", str(min_time.total_seconds()), min_rmse, "kNN", best_k_reg, "Regression", outlier_reg_knn_min, mse_reg_knn_min],  
        ['Max', "", str(max_time.total_seconds()), max_rmse, "kNN", best_k_reg, "Regression", outlier_reg_knn_max, mse_reg_knn_max],  
        ['Full', "", "", "", "kNN", best_k_reg, "Regression", outlier_reg_knn_full, mse_reg_knn_full],  

        ['kNN', knn_impute_k, str(knn_time.total_seconds()), knn_rmse, "kNN", best_k_class, "Classification", outlier_class_knn_knn, mse_class_knn_knn],  
        ['MICE', "", str(mice_time.total_seconds()), mice_rmse, "kNN", best_k_class, "Classification", outlier_class_knn_mice, mse_class_knn_mice],  
        ['GAIN', "", str(gain_time.total_seconds()), gain_rmse, "kNN", best_k_class, "Classification", outlier_class_knn_gain, mse_class_knn_gain],  
        ['Mean', "", str(mean_time.total_seconds()), mean_rmse, "kNN", best_k_class, "Classification", outlier_class_knn_mean, mse_class_knn_mean],  
        ['Median', "", str(median_time.total_seconds()), median_rmse, "kNN", best_k_class, "Classification", outlier_class_knn_median, mse_class_knn_median],  
        ['Min', "", str(min_time.total_seconds()), min_rmse, "kNN", best_k_class, "Classification", outlier_class_knn_min, mse_class_knn_min],  
        ['Max', "", str(max_time.total_seconds()), max_rmse, "kNN", best_k_class, "Classification", outlier_class_knn_max, mse_class_knn_max],  
        ['Full', "", "", "", "kNN", best_k_class, "Classification", outlier_class_knn_full, mse_class_knn_full], 

        ['kNN', knn_impute_k, str(knn_time.total_seconds()), knn_rmse, "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_knn, mse_reg_rf_knn],  
        ['MICE', "", str(mice_time.total_seconds()), mice_rmse, "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_mice, mse_reg_rf_mice],  
        ['GAIN', "", str(gain_time.total_seconds()), gain_rmse, "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_gain, mse_reg_rf_gain],  
        ['Mean', "", str(mean_time.total_seconds()), mean_rmse, "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_mean, mse_reg_rf_mean],  
        ['Median', "", str(median_time.total_seconds()), median_rmse, "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_median, mse_reg_rf_median],  
        ['Min', "", str(min_time.total_seconds()), min_rmse, "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_min, mse_reg_rf_min],  
        ['Max', "", str(max_time.total_seconds()), max_rmse, "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_max, mse_reg_rf_max], 
        ['Full', "", "", "", "RandomForest", best_tree_reg, "Regression", outlier_reg_rf_full, mse_reg_rf_full],

        ['kNN', knn_impute_k, str(knn_time.total_seconds()), knn_rmse, "RandomForest", best_tree_class, "Classification", outlier_class_rf_knn, mse_class_rf_knn],  
        ['MICE', "", str(mice_time.total_seconds()), mice_rmse, "RandomForest", best_tree_class, "Classification", outlier_class_rf_mice, mse_class_rf_mice],  
        ['GAIN', "", str(gain_time.total_seconds()), gain_rmse, "RandomForest", best_tree_class, "Classification", outlier_class_rf_gain, mse_class_rf_gain],  
        ['Mean', "", str(mean_time.total_seconds()), mean_rmse, "RandomForest", best_tree_class, "Classification", outlier_class_rf_mean, mse_class_rf_mean],  
        ['Median', "", str(median_time.total_seconds()), median_rmse, "RandomForest", best_tree_class, "Classification", outlier_class_rf_median, mse_class_rf_median],  
        ['Min', "", str(min_time.total_seconds()), min_rmse, "RandomForest", best_tree_class, "Classification", outlier_class_rf_min, mse_class_rf_min],  
        ['Max', "", str(max_time.total_seconds()), max_rmse, "RandomForest", best_tree_class, "Classification", outlier_class_rf_max, mse_class_rf_max],  
        ['Full', "", "", "", "RandomForest", best_tree_class, "Classification", outlier_class_rf_full, mse_class_rf_full]]

    error_frame = pd.DataFrame(error_lists, columns=["FillMethod", "FillKVal", "FillTime", "FillRMSE", "PredictMethod", "BestK_TreeVal", "Class/Regress", "OutlierRate", "MSE/Accuracy"])
    
    startTime = datetime.now()
    perc_folder = "Missing_Rate-" + str(missing_percentage)
    if not os.path.exists("Results"):
        os.makedirs("Results")
    os.chdir("Results")
    if not os.path.exists(perc_folder):
        os.makedirs(perc_folder)
    os.chdir(perc_folder)
    folderpath = startTime.strftime("%d-%m-%Y_%H-%M-%S") + "_Seed-" + str(initial_rand_seed)   
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    os.chdir(folderpath)
    
    error_frame.to_csv("summary.csv", index=False)
    np.savetxt('y_vals_test.csv', y_vals_test, fmt="%f", delimiter=",")
    np.savetxt('y_vals_class_test.csv', y_vals_class_test, fmt="%f", delimiter=",")

    np.savetxt('pred_reg_knn_knn.csv', pred_reg_knn_knn, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_knn.csv', pred_class_knn_knn, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_knn.csv', pred_reg_rf_knn, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_knn.csv', pred_class_rf_knn, fmt="%f", delimiter=",")

    plot_data(y_vals_test, pred_reg_knn_knn, file_name="pred_reg_knn_knn.pdf")
    plot_data(y_vals_class_test, pred_class_knn_knn, file_name="pred_class_knn_knn.pdf")
    plot_data(y_vals_test, pred_reg_rf_knn, file_name="pred_reg_rf_knn.pdf")
    plot_data(y_vals_class_test, pred_class_rf_knn, file_name="pred_class_rf_knn.pdf")

    np.savetxt('pred_reg_knn_mice.csv', pred_reg_knn_mice, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_mice.csv', pred_class_knn_mice, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_mice.csv', pred_reg_rf_mice, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_mice.csv', pred_class_rf_mice, fmt="%f", delimiter=",")
    
    plot_data(y_vals_test, pred_reg_knn_mice, file_name="pred_reg_knn_mice.pdf")
    plot_data(y_vals_class_test, pred_class_knn_mice, file_name="pred_class_knn_mice.pdf")
    plot_data(y_vals_test, pred_reg_rf_mice, file_name="pred_reg_rf_mice.pdf")
    plot_data(y_vals_class_test, pred_class_rf_mice, file_name="pred_class_rf_mice.pdf")

    np.savetxt('pred_reg_knn_gain.csv', pred_reg_knn_gain, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_gain.csv', pred_class_knn_gain, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_gain.csv', pred_reg_rf_gain, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_gain.csv', pred_class_rf_gain, fmt="%f", delimiter=",")
    
    plot_data(y_vals_test, pred_reg_knn_gain, file_name="pred_reg_knn_gain.pdf")
    plot_data(y_vals_class_test, pred_class_knn_gain, file_name="pred_class_knn_gain.pdf")
    plot_data(y_vals_test, pred_reg_rf_gain, file_name="pred_reg_rf_gain.pdf")
    plot_data(y_vals_class_test, pred_class_rf_gain, file_name="pred_class_rf_gain.pdf")

    np.savetxt('pred_reg_knn_mean.csv', pred_reg_knn_mean, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_mean.csv', pred_class_knn_mean, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_mean.csv', pred_reg_rf_mean, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_mean.csv', pred_class_rf_mean, fmt="%f", delimiter=",")
    
    plot_data(y_vals_test, pred_reg_knn_mean, file_name="pred_reg_knn_mean.pdf")
    plot_data(y_vals_class_test, pred_class_knn_mean, file_name="pred_class_knn_mean.pdf")
    plot_data(y_vals_test, pred_reg_rf_mean, file_name="pred_reg_rf_mean.pdf")
    plot_data(y_vals_class_test, pred_class_rf_mean, file_name="pred_class_rf_mean.pdf")

    np.savetxt('pred_reg_knn_median.csv', pred_reg_knn_median, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_median.csv', pred_class_knn_median, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_median.csv', pred_reg_rf_median, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_median.csv', pred_class_rf_median, fmt="%f", delimiter=",")
    
    plot_data(y_vals_test, pred_reg_knn_median, file_name="pred_reg_knn_median.pdf")
    plot_data(y_vals_class_test, pred_class_knn_median, file_name="pred_class_knn_median.pdf")
    plot_data(y_vals_test, pred_reg_rf_median, file_name="pred_reg_rf_median.pdf")
    plot_data(y_vals_class_test, pred_class_rf_median, file_name="pred_class_rf_median.pdf")

    np.savetxt('pred_reg_knn_min.csv', pred_reg_knn_min, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_min.csv', pred_class_knn_min, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_min.csv', pred_reg_rf_min, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_min.csv', pred_class_rf_min, fmt="%f", delimiter=",")
    
    plot_data(y_vals_test, pred_reg_knn_min, file_name="pred_reg_knn_min.pdf")
    plot_data(y_vals_class_test, pred_class_knn_min, file_name="pred_class_knn_min.pdf")
    plot_data(y_vals_test, pred_reg_rf_min, file_name="pred_reg_rf_min.pdf")
    plot_data(y_vals_class_test, pred_class_rf_min, file_name="pred_class_rf_min.pdf")

    np.savetxt('pred_reg_knn_max.csv', pred_reg_knn_max, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_max.csv', pred_class_knn_max, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_max.csv', pred_reg_rf_max, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_max.csv', pred_class_rf_max, fmt="%f", delimiter=",")
    
    plot_data(y_vals_test, pred_reg_knn_max, file_name="pred_reg_knn_max.pdf")
    plot_data(y_vals_class_test, pred_class_knn_max, file_name="pred_class_knn_max.pdf")
    plot_data(y_vals_test, pred_reg_rf_max, file_name="pred_reg_rf_max.pdf")
    plot_data(y_vals_class_test, pred_class_rf_max, file_name="pred_class_rf_max.pdf")

    np.savetxt('pred_reg_knn_full.csv', pred_reg_knn_full, fmt="%f", delimiter=",")
    np.savetxt('pred_class_knn_full.csv', pred_class_knn_full, fmt="%f", delimiter=",")
    np.savetxt('pred_reg_rf_full.csv', pred_reg_rf_full, fmt="%f", delimiter=",")
    np.savetxt('pred_class_rf_full.csv', pred_class_rf_full, fmt="%f", delimiter=",")
    
    plot_data(y_vals_test, pred_reg_knn_full, file_name="pred_reg_knn_full.pdf")
    plot_data(y_vals_class_test, pred_class_knn_full, file_name="pred_class_knn_full.pdf")
    plot_data(y_vals_test, pred_reg_rf_full, file_name="pred_reg_rf_full.pdf")
    plot_data(y_vals_class_test, pred_class_rf_full, file_name="pred_class_rf_full.pdf")


    np.savetxt('x_vals_mice.csv', x_vals_mice, fmt="%f", delimiter=",")
    np.savetxt('x_vals_gain.csv', x_vals_gain, fmt="%f", delimiter=",")
    np.savetxt('x_vals_knn.csv', x_vals_knn, fmt="%f", delimiter=",")
    np.savetxt('x_vals_mean.csv', x_vals_mean, fmt="%f", delimiter=",")
    np.savetxt('x_vals_median.csv', x_vals_median, fmt="%f", delimiter=",")
    np.savetxt('x_vals_max.csv', x_vals_max, fmt="%f", delimiter=",")
    np.savetxt('x_vals_min.csv', x_vals_min, fmt="%f", delimiter=",")

    np.savetxt('missing_mask.csv', missing_mask, fmt="%f", delimiter=",")
    np.savetxt('x_vals_test.csv', x_vals_test, fmt="%f", delimiter=",")

    np.savetxt('bin_edges.csv', bin_edges, fmt="%f", delimiter=",")
    np.savetxt('bin_median.csv', bin_median, fmt="%f", delimiter=",")
     


if __name__ == "__main__":
	main()