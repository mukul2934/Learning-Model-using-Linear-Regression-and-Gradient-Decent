import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
from pylab import *
from scipy import *

def gradient_decent(alpha, x, y, ep = 0.0001, max_iter = 10000):
    converged = False
    itr = 0;
    m = x.shape[0]

    # Choose initial thetas to be random
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    # total error using Squared Error Model
    J = sum([(t0 + t1 * x[i] - y[i])**2 for i in range(m)])

    # Determine optimal thetas
    while not converged:
        # for each training sample calulate the gradient
        grad0 = 1.0/m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])

        # calulate new thetas
        tmp0 = t0 - alpha * grad0
        tmp1 = t1 - alpha * grad1

        # update the thetas
        t0 = tmp0
        t1 = tmp1

        # Compute the mean squared error
        e = sum([(t0 + t1 * x[i] - y[i])**2 for i in range(m)])

        if abs(J - e) <= ep:
            print "Convered, iterations: ", itr
            converged = True

        J = e # update the error value
        itr += 1

        if itr == max_iter:
            print "Max iterations reached"
            converged = True

    return t0, t1

if __name__ == "__main__":
    # Create sample regression data
    x, y = make_regression(n_samples = 500, n_features = 1, n_informative = 1, random_state = 0, noise = 35)
    print "x.shape = %s y.shape = %s" %(x.shape, y.shape)

    alpha = 0.01 # learning rate
    ep = 0.01 # convergence criteria

    # call the gradient decent function to determine the optimum theta_0 and theta_1 values
    theta_0, theta_1 = gradient_decent(alpha, x, y, ep, max_iter = 1000)
    print "theta_0 = %s, theta_1 = %s" %(theta_0, theta_1)

    # plot
    for i in range(x.shape[0]):
        y_predict = theta_0 + theta_1 * x

    plot(x, y, 'o')
    plot(x, y_predict, 'k-')
    show()
    print "Done!"
