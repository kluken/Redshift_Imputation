# from astropy.table import Table
from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def read_fits(filename, colnames):
    """Read an astronomical table, outputting specific columns. 

    Args:
        filename (String): Name of file to be read
        colnames ([String]): Array of columns to be read. 

    Returns:
        table_data (Astropy Table): Astropy table of data
    """
    # table_data = Table.read(filename)

    hdul = fits.open(filename)
    hdulData = hdul[1].data
    #Create catalogueData array from the redshift column
    table_data = np.reshape(np.array(hdulData.field(colnames[0]), dtype=np.float32), [len(hdulData.field(colnames[0])),1])
    #Add the columns required for the test
    for i in range(1, len(colnames)):
        table_data = np.hstack([table_data,np.reshape(np.array(hdulData.field(colnames[i]), dtype=np.float32), [len(hdulData.field(colnames[i])),1])])
    return table_data


def split_data(x_vals, y_vals, rand_generator=None, field_one=None, field_two=None, field_list=None):
    """Splits large data set into training and test sets.

    Args:
        x_vals (Numpy Array (2-D)): 2-D Array holding the x-vals
        y_vals (Numpy Array (1-D)): 1-D Array holding the y-vals
        rand_generator (Numpy Random Generator, optional): Random Generator to use to generate 70-30 Training-Testing Split. Defaults to None.
        field_one (String, optional): Field to use as the training set. Defaults to None.
        field_two (String, optional): Field to use as the test set. Defaults to None.
        field_list (Numpy Array (1-D), optional): Array holding the field each galaxy belongs in. Defaults to None.

    Returns:
        Numpy Arrays: The split Training and Test sets. 
    """
    if rand_generator is not None:
        test_indices = rand_generator.choice(len(x_vals), round(len(x_vals)*0.3), replace=False)
        train_indices = np.array(list(set(range(len(x_vals))) - set(test_indices)))
        x_vals_train = x_vals[train_indices]
        x_vals_test = x_vals[test_indices]
        y_vals_train = y_vals[train_indices]
        y_vals_test = y_vals[test_indices]
    else:
        x_vals_test = x_vals[np.where(field_list == field_one)[0]]
        y_vals_test = y_vals[np.where(field_list == field_one)[0]]
        x_vals_train = x_vals[np.where(field_list == field_two)[0]]
        y_vals_train = y_vals[np.where(field_list == field_two)[0]]

    return x_vals_train, x_vals_test, y_vals_train, y_vals_test

def norm_mad(data, axis=None):
    """Calculate the Median Absolute Deviation

    Args:
        data ([float]): Array of values to calculate the MAD of. 
        axis (int, optional): Which axis (if using 2+-d Array). Defaults to None.

    Returns:
        Median Absolute Deviation, calculated over the provided axis
    """
    return (1.4826 * np.median(np.absolute(data - np.median(data, axis)), axis))

def outlier_rate(resid, out_frac=0.15):
    """Calculate the outlier rate fraction

    Args:
        resid (array): Array of residuals, normalised by redshift. 
        out_frac (decimal): Value to use as the outlier cutoff. 

    Returns:
        [type]: [description]
    """
    outlier=100*len(resid[np.where(abs(resid)>out_frac)])/len(resid)
    return outlier

def norm_std_dev(resid):
    """Calculate the normalised standard deviation

    Args:
        resid (array): Array of residuals, normalised by redshift. 
        out_frac (decimal): Value to use as the outlier cutoff. 

    Returns:
        [type]: [description]
    """
    sigma=np.std(resid)
    return sigma

def norm_residual(spec_z, pred_z):
    """Calculate the outlier rate fraction

    Args:
        spec_z (array): Measured redshifts
        pred_z (array): Predicted redshifts
        out_frac (decimal): Value to use as the outlier cutoff. 

    Returns:
        [type]: [description]
    """
    residual=(spec_z-pred_z)/(1+spec_z)
    return residual

def plot_data(spec_z, pred_z, file_name=None, error = None, kVal = None):
    """Plot the results. 

    Args:
        spec_z (np.array): Array of measured redshifts. 
        pred_z (np.array): Array of spectroscopic redshifts. 
        fileName (String): Filename to save the plot to. 
        error (bool, optional): Error bars asssociated with the predicted redshifts. Defaults to False.
    """

    # Calculate the different stats to plot.
    residual=norm_residual(spec_z,pred_z) 
    out_num = outlier_rate(residual, 0.15)
    sigma=norm_std_dev(residual)
    nmad=norm_mad(residual)
    
    fig, [ax,ay] = plt.subplots(2, sharex=True,  gridspec_kw = {'height_ratios':[2, 1]})
    fig.set_figheight(9)
    fig.set_figwidth(6)
    sizeElem=2
    if error is None:
        # If there are no errors plot, just plot the measured vs predicted, and 
        # measured vs residual
        cax=ax.scatter(spec_z, pred_z, edgecolor='face', s=sizeElem, color="black")
        cay=ay.scatter(spec_z, residual, edgecolor='face', s=sizeElem, color="black")
    else:
        # Else, plot the same things, with error bars. First plots transparent error bars, second plots solid markers. 
        cax=ax.errorbar(spec_z, pred_z, yerr = error, color="black", ms = sizeElem, lw = 1, fmt="none", alpha=0.2)
        cax=ax.scatter(spec_z, pred_z, edgecolor=None, s=sizeElem, color="black")
        cay=ay.scatter(spec_z, residual, edgecolor=None, s=sizeElem, color="black")

    # Plot the guide lines        
    ax.plot([0,4],[0,4], 'r--',linewidth=1.5)
    ax.plot([0,4],[0.15,4.75], 'b--',linewidth=1.5)
    ax.plot([0,4],[-.15,3.25], 'b--',linewidth=1.5)
    ay.plot([0,4],[0,0], 'r--',linewidth=1.5)
    ay.plot([0,4],[0.15,.15], 'b--',linewidth=1.5)
    ay.plot([0,4],[-.15,-.15], 'b--',linewidth=1.5)
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    ax.axis([0,4,0, 4.6])
    ay.axis([0,4,-.5, .5])

    xlab=.3
    ylab=3.7
    step=-.3
    ax.text(xlab, ylab, r'$N='+str(spec_z.shape[0])+'$')
    ax.text(xlab, ylab+ step, r'$\sigma='+str(round(sigma, 2))+r'$')
    ax.text(xlab, ylab+ 2*step,        r'$NMAD='+str(round(nmad, 2))+r'$')
    ax.text(xlab, ylab+ 3*step,        r'$\eta='+str(round(out_num, 2))+r'\%$')
    if (kVal is not None):
        ax.text(xlab, ylab+ 4*step,        r'$k='+str(kVal)+r'$')
    ax.set_ylabel('$z_{photo}$')
    ay.set_ylabel(r'$\frac{z_{spec}-z_{photo}}{z_{spec}+1}$')
    ax.set_xlabel('$z_{spec}$')
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)
        plt.clf()
    else:
        plt.show()


def kNN(kVal, xValsTrain, xValsTest, yValsTrain, yValsTest, distType):
    """Run kNN regression

    Args:
        kVal (int): Value to use as k for kNN
        xValsTrain (np.array): 2-d np.array holding the photometry used for training
        xValsTest (np.array): 2-d np.array holding the photometry used for testing
        yValsTrain (np.array): 1-d np.array holding the measured redshift for training
        yValsTest (np.array): 1-d np.array holding the measured redshift for testing
        distType ([type]): Integer used to determine the distance metric. If less than 5, minkowski distance used. If more, mahalanobis. 

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    if distType < 5:
        neigh = KNeighborsRegressor(n_neighbors=kVal, p=distType)
    elif distType == 99:   
        neigh = KNeighborsRegressor(n_neighbors=kVal, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain, rowvar=False)})
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest))


def kNN_classification(kVal, xValsTrain, xValsTest, yValsTrain, yValsTest, distType):
    """Run kNN classification. 

    Args:
        kVal (int): Value to use as k for kNN
        xValsTrain (np.array): 2-d np.array holding the photometry used for training
        xValsTest (np.array): 2-d np.array holding the photometry used for testing
        yValsTrain (np.array): 1-d np.array holding the measured redshift for training
        yValsTest (np.array): 1-d np.array holding the measured redshift for testing
        distType ([type]): Integer used to determine the distance metric. If less than 5, minkowski distance used. If more, mahalanobis. 

    Returns:
        np.array: 1-d np.array holding the predictions
        float: accuracy of the predictions
    """
    if distType < 5:
        neigh = KNeighborsClassifier(n_neighbors = kVal, p = distType)
    elif distType == 99:
        neigh = KNeighborsClassifier(n_neighbors = kVal, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain, rowvar=False)})
    neigh.fit(xValsTrain,np.squeeze(yValsTrain.astype(str)))
    predictions = neigh.predict(xValsTest.astype(str))
    return predictions.astype(np.float32).ravel(), neigh.score(xValsTest,np.squeeze(yValsTest.astype(str))).astype(np.float32)


def randomForestRegress(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, randomState=None):
    """Run random forest regression

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    if randomState is None:
        randomState = 42
    neigh = RandomForestRegressor(treeVal, random_state=randomState)
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest))


def randomForestClass(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, randomState = None):
    """Run Random Forest Classification

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: Predicted redshifts. 
        float: Prediction accuracies
    """
    if randomState is None:
        randomState = 42
    neigh = RandomForestClassifier(treeVal, random_state=randomState)
    neigh.fit(xValsTrain,np.squeeze(yValsTrain.astype(str)))
    predictions = neigh.predict(xValsTest.astype(str))
    return predictions.astype(np.float32).ravel(), neigh.score(xValsTest,np.squeeze(yValsTest.astype(str))).astype(np.float32)

def binDataFunc(redshiftVector, numBins, maxRedshift = 1.5):
    """Function to bin the data

    Args:
        redshiftVector (np.array): 1-d numpy array holding the redshifts to be binned
        numBins (integer): Number of bins to use
        maxRedshift (float, optional): Value to use as the highest bin edge. Defaults to 1.5.

    Returns:
        np.array: List containing the binned redshifts
        np.array: List containing each of the bin edges
        np.array: List containing the centres of the bins
    """
    sortedRedshift = np.sort(redshiftVector, axis=None)
    
    numPerBin = sortedRedshift.shape[0]//numBins #Integer division!
    # Set first bin edge to be the lowest value supplied
    binEdges = [0]
    # Find each of the bin edges
    for i in range(1, numBins):
        binEdges.append(i * numPerBin)
    binEdges.append(sortedRedshift.shape[0]-1)
    # Replace the indices of the bin edges with the bin edge values
    binEdges = sortedRedshift[binEdges]
    binEdges[-1] = maxRedshift
    
    # New list to hold the median of each bins
    newZ = []
    for i in range(1, numBins + 1):
        if i < numBins:
            newZ.append(np.median([binEdges[i-1], binEdges[i]]))
        else:
            newZ.append(np.median(redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < np.max(sortedRedshift)))[0]])) 
    # Bin the data
    for i in range(1, numBins + 1):
        if i < numBins:
            if i == 1:
                redshiftVector[np.where((redshiftVector < binEdges[i]))[0]] = newZ[i-1]
            else:
                redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < binEdges[i]))[0]] = newZ[i-1]
        else: 
            redshiftVector[np.where((redshiftVector >= binEdges[i-1]))[0]] = newZ[i-1]
    return redshiftVector, binEdges, newZ


def norm_x_vals(x_vals_train, x_vals_test):
    """Normalise x_vals, based on the training sample. 

    Args:
        x_vals_train (np.array): x_vals training sample
        x_vals_test (np.array): x_vals test sample

    Returns:
        np.array: Normalised x_vals_train
        np.array: Normalised y_vals_train
        np.array: 1-d array with the mean of each feature
        np.array: 1-d array with the std dev of each feature
        
    """
    mean_norm = np.mean(x_vals_train, axis = 0)
    std_norm = np.std(x_vals_train, axis = 0)
    x_vals_train = (x_vals_train - mean_norm) / std_norm
    x_vals_test = (x_vals_test - mean_norm) / std_norm

    return x_vals_train, x_vals_test, mean_norm, std_norm


data = read_fits("ATLAS_Complete.fits", ["z","Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"])

from numpy.random import default_rng
from sklearn.impute import KNNImputer
rand_generator = default_rng(seed=42)
x_vals = data[:,1:]

test_indices = rand_generator.integers(0, len(x_vals), size = round(len(x_vals)*(1-0.7)))

train_indices = np.array(list(set(range(len(x_vals))) - set(test_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
print(x_vals_train.shape, x_vals_test.shape)

from gain_utils import blank_data
x_vals_test_blank, mask = blank_data(x_vals_test, 0.0001, rand_generator)
from gain_utils import rmse_loss

impute = KNNImputer(n_neighbors=2)
impute.fit(x_vals_train)
x_vals_test_blank = impute.transform(x_vals_test_blank)

print(rmse_loss(x_vals_test,x_vals_test_blank, mask))