# Redshift Imputation

## Requirements:
* See Conda_Env.yml for packages used. Can be installed using the `conda env create --file Conda_Env.yml` command after changing the name and prefix to the desired values.

## Usage:
* Using arguments:
  * -s | --seed. Initial random seed to use. 
  * -m | --missingRate. The percentage of data to set to missing. 
  * -t | --tqdmDisable. Boolean, use to turn off the TQDM Statusbar if running multiple jobs. 
 
 ## Examples:
 * `Imputation.py -s 42 -m 5 -t`
  * Using this will run the Imputation script, using an initial seed of 42, setting 5% of the data to be missing and disabling the TQDM statusbar
