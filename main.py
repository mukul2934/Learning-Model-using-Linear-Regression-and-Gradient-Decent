import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
from pylab import *
from scipy import *

if __name__ == "__main__":
    x, y = make_regression(n_samples = 100, n_features = 1, n_informative = 1, random_state = 0, noise = 35)
    print "x.shape = %s y.shape = %s" %(x.shape, y.shape)

    plot(x,y)
    show()
