from sklearn.impute import KNNImputer
from sklearn import preprocessing
import timeit
from math import ceil
from gain_utils import rmse_loss

def kNNImpute(rand_generator, x_vals, x_vals_blanked, blank_map, train_test = 0.7):
        test_indices = rand_generator.integers(0, len(x_vals), size = round(len(x_vals)*(1-train_test)))

        train_indices = np.array(list(set(range(len(x_vals))) - set(test_indices)))
        x_vals_train = x_vals[train_indices]
        x_vals_test = x_vals[test_indices]
