import numpy as np
import scipy.io  # Use scipy.io to import the data
np.random.seed(1234)
# First, load the data sets as numpy arrays
mat_data_train = scipy.io.loadmat('./sarcos_inv.mat')
mat_data_test = scipy.io.loadmat('./sarcos_inv_test.mat')
data_train = mat_data_train['sarcos_inv']
data_test = mat_data_test['sarcos_inv_test']

# Then, split it appropriately in the following variables
data = data_train

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

shuffled_data = data[indices]

split_point = 35587

train_data = shuffled_data[:split_point]
val_data = shuffled_data[split_point:]

# Input and output training data
xs_train = train_data[:, :21]
ys_train = train_data[:, 21:]

# Input and output validation data
xs_valid = val_data[:, :21]
ys_valid = val_data[:, 21:]

# Input and output test data
xs_test = data_test[:, :21]
ys_test = data_test[:, 21:]

print( "xs_train", xs_train.shape )
print( "ys_train", ys_train.shape )
print( "xs_valid", xs_valid.shape )
print( "ys_valid", ys_valid.shape )
print( "xs_test", xs_test.shape )
print( "ys_test", ys_test.shape )