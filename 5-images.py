# Import necessary modules
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
#print(digits['DESCR'])

# Print the shape of the images and data keys
print(digits['images'].shape)
print(digits['data'].shape)

# Display digit 1010
print(plt)
plt.imshow(digits['images'][1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()