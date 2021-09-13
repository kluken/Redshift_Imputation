import sys, argparse, pickle, os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

from sklearn.model_selection import KFold 
from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    # TODO: Make these arguments to be passed. 

    initial_rand_seed = 42 # argument
    initial_rand_gen = np.random.default_rng(initial_rand_seed) # Probably removed from this file and moved to a controller script
    split_data_seed = initial_rand_gen.integers(314159265) # Will be argument
    missing_data_seed = initial_rand_gen.integers(314159265) # Will be argument
    mice_seed = initial_rand_gen.integers(314159265) # Will be argument
    gain_seed = initial_rand_gen.integers(314159265) # Will be argument
    knn_seed = initial_rand_gen.integers(314159265) # Will be argument
    tree_seed = initial_rand_gen.integers(314159265) # Will be argument
    missing_percentage = 0.02 # Will be argument, of values 2 5 10 15 20 25 30
    knn_distance = 99 # Mahalanobis. Can be swapped later if needed, but will probably stay. 1 for Manhattan, 2 for Euclidean, 99 for Mahalanobis
    regression_test = True # Will be argument
    outlier_fail_rate = 0.15 # Value to use for the Outlier Rate
    num_class_bins = 15 # Number of bins to use for classification if needed
    kfold_splits = 10 # Used in k-Fold Cross Validation.
    catalogue = "../Data/ATLAS_Complete_fixed.fits" # Dataset to start with use. Will probably be argument
    data_cols = ["z","Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"] # Columns to use. Might be argument
    k_range = np.arange(3, 33, 2)
    tree_range = np.arange(1, 61, 1)


    full_dataset = read_fits(catalogue, data_cols)
    x_vals = full_dataset[:, 1:]
    y_vals = full_dataset[:,0]

    y_vals_class, bin_edges, bin_median = bin_data_func(y_vals, num_class_bins)

    x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(x_vals, y_vals, np.random.default_rng(split_data_seed))
    _, _, y_vals_class_train, y_vals_class_test = split_data(x_vals, y_vals_class, np.random.default_rng(split_data_seed))

    x_vals_blank, missing_mask = blank_data(x_vals_test, missing_percentage, np.random.default_rng(missing_data_seed))

    x_vals_knn = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_knn, knn_impute_k = kNN_impute(x_vals_knn, x_vals_train, np.random.default_rng(knn_seed), missing_percentage)
    knn_time = start_time - datetime.now()
    knn_rmse = rmse_loss(x_vals_test, x_vals_knn, missing_mask)

    start_time = datetime.now()
    mice_obj = FastMICE(rand_generator=np.random.default_rng(mice_seed), tqdm_disable=False)
    x_vals_mice = pd.DataFrame(x_vals_blank)
    x_vals_mice = np.array(mice_obj.fill_missing_values(x_vals_mice))
    mice_time = start_time - datetime.now()
    mice_rmse = rmse_loss(x_vals_test, x_vals_mice, missing_mask)

    gain_parset = {'batch_size': 128,
                    'hint_rate': 0.9,
                    'alpha': 100,
                    'iterations': 60000}
    start_time = datetime.now()
    #  Default gain parameters used by Rabina were: batch size = 128, hint rate = 0.9, alpha = 100, iterations = 60000.
    x_vals_gain = gain(x_vals_blank, gain_parset, np.random.default_rng(gain_seed), tqdm_disable=False)
    gain_time = datetime.now()
    gain_rmse = rmse_loss(x_vals_test, x_vals_gain, missing_mask)
    

    x_vals_mean = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_mean = simple_impute(x_vals_mean, np.nanmean, np.nan)
    mean_time = start_time - datetime.now()
    mean_rmse = rmse_loss(x_vals_test, x_vals_mean, missing_mask)


    x_vals_min = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_min = simple_impute(x_vals_min, np.nanmin, np.nan)
    min_time = start_time - datetime.now()
    min_rmse = rmse_loss(x_vals_test, x_vals_min, missing_mask)

    x_vals_max = np.copy(x_vals_blank)
    start_time = datetime.now()
    x_vals_max = simple_impute(x_vals_max, np.nanmax, np.nan)
    max_time = start_time - datetime.now()
    max_rmse = rmse_loss(x_vals_test, x_vals_max, missing_mask)



    # Collection Arrays
    outlier_knn_regress = []
    outlier_knn_class = []
    outlier_random_regress = []
    outlier_random_class = []
    mse_knn_regress = []
    acc_knn_class = []
    mse_random_regress = []
    acc_random_class = []

    for k in tqdm(k_range):
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
            pred_reg, mse_reg = kNN(k, x_vals_train_norm, x_vals_test_norm, y_vals_cross_train, y_vals_cross_test, knn_distance)
            
            #predict y_vals for classification
            pred_class, acc_class = kNN_classification(k, x_vals_train_norm, x_vals_test_norm, y_vals_cross_train_class, y_vals_cross_test_class, knn_distance)

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
    for tree in tqdm(tree_range):
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

    best_k_reg = k_range[np.argmin(outlier_knn_regress)]
    best_k_class = k_range[np.argmin(outlier_knn_class)]
    best_tree_reg = tree_range[np.argmin(outlier_random_regress)]
    best_tree_class =  tree_range[np.argmin(outlier_random_class)]

    # Normalise x_vals

    # knn regression
    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_knn)
    pred_reg_knn_knn, mse_reg_knn_knn = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_knn, mse_class_knn_knn = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_reg_rf_knn, mse_reg_rf_knn = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_knn, mse_class_rf_knn = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_mice)
    pred_reg_knn_mice, mse_reg_knn_mice = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_mice, mse_class_knn_mice = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_reg_rf_mice, mse_reg_rf_mice = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_mice, mse_class_rf_mice = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_gain)
    pred_reg_knn_gain, mse_reg_knn_gain = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_gain, mse_class_knn_gain = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_reg_rf_gain, mse_reg_rf_gain = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_gain, mse_class_rf_gain = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_mean)
    pred_reg_knn_mean, mse_reg_knn_mean = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_mean, mse_class_knn_mean = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_reg_rf_mean, mse_reg_rf_mean = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_mean, mse_class_rf_mean = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_min)
    pred_reg_knn_min, mse_reg_knn_min = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_min, mse_class_knn_min = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_reg_rf_min, mse_reg_rf_min = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_min, mse_class_rf_min = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)

    x_vals_train_norm, x_vals_test_norm, _, _ = norm_x_vals(x_vals_train, x_vals_max)
    pred_reg_knn_max, mse_reg_knn_max = kNN(best_k_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_class_knn_max, mse_class_knn_max = kNN_classification(best_k_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, knn_distance)
    pred_reg_rf_max, mse_reg_rf_max = random_forest_regress(best_tree_reg, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    pred_class_rf_max, mse_class_rf_max = random_forest_class(best_tree_class, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, rf_seed)
    
    print(datetime.now() - starting_time)

    """
    Loop - 10 (100?) iterations, using different random seeds to split data 70/30. Loop structure isn't actually implemented - this would be part of the multiprocessing section
        Blank X% of test data
        if classification - bin  redshift data
        Impute with all methods, giving 5-6 test sets

        Loop - 10 fold cross validation 
            Optimise parameters - k for knn or trees for rf
            Predict!




    """



if __name__ == "__main__":
	main()