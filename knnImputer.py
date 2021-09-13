from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold 
import numpy as np
from tqdm import tqdm
from gain_utils import rmse_loss, blank_data

def kNN_impute(x_vals_blanked, x_vals_train, rand_seed, miss_perc, tqdm_disable=False):
    rand_gen = np.random.default_rng(rand_seed)
    x_vals_train_blank, missing_mask = blank_data(x_vals_train, miss_perc, rand_gen)
    k_range = np.arange(1, 31, 2)
    best_rmse = []
    for i in tqdm(k_range, disable=tqdm_disable):
        imputer = KNNImputer(n_neighbors=i)
        x_imputed = imputer.fit_transform(x_vals_train_blank)
        loss = rmse_loss(x_vals_train, x_imputed, missing_mask)
        best_rmse.append(loss)
    best_k = k_range[np.argmin(best_rmse)]
    imputer = KNNImputer(n_neighbors=best_k)
    x_imputed = imputer.fit_transform(x_vals_blanked)
    return x_imputed, best_k
