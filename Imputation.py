



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