from multiprocessing import Pool
import os
import numpy as np

missing_rates = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
rand_gen = np.random.default_rng(42)
tests = []

for miss_rate in missing_rates:
    for i in range(101):
        random_seed = rand_gen.integers(314159265)
        test_string = "python Imputation.py -s " + str(random_seed) + " -m " + str(miss_rate) + " -t"
        tests.append(test_string)

def processTests(test):
    os.system(test)

if __name__ == '__main__':
    pool = Pool(os.cpu_count())       # Create a multiprocessing Pool
    pool.map(processTests, tests) # process data_inputs iterable with pool
    pool.close()
    pool.join()