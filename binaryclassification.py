import numpy as np
import pandas as pd

data_train = pd.read_csv('train_and_test2.csv', delimiter=',', index_col='Passengerid')

print(data_train.head())