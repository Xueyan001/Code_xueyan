
# example of plotting a histogram of a random sample
from matplotlib import pyplot
from numpy.random import normal
import numpy as np

# generate a sample
#sample = normal(size=1000)
sample = np.random.normal(loc=0, scale=1, size=1000) #loc:mean scale:standard deviation size: sample count
# plot a histogram of the sample
pyplot.hist(sample, bins=10)
pyplot.show()

pyplot.hist(sample, bins=3)
pyplot.show()