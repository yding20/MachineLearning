import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

Data = np.loadtxt('points.dat')
X = Data[:,0]
Y = Data[:,1]

def TwoDGaussian(xinput, mean, cov):
    detcov = np.linalg.det(cov)
    temp = np.dot(np.transpose(xinput - mean), np.linalg.inv(cov))
    Epart = np.exp(-0.5*np.dot(temp, (xinput - mean)))
    return 1/(2*np.pi*np.sqrt(detcov))*Epart

# Initilize the means mu, covariances sigma and mix coefficient pi
# and log likelihood, Test 4 clusters case.

mu =  np.array([[-2, 2], [2, 2],  [-2, -2], [2, -2]])
sigma = np.zeros((4,2,2))
sigma[:,0,0] = 1
sigma[:,1,1] = 1
pi = [0.25, 0.25, 0.25, 0.25]
gamma = np.zeros((len(X), len(pi)))
gammanominator = np.zeros((len(X), len(pi)))

for n in range(0, len(X)):
    likelihood = 0
    loglikelihood = 0
    xinput = np.array([X[n],Y[n]])
    for k in range(0, len(pi)):
        gauss = TwoDGaussian(xinput, mu[k,:], sigma[k,:,:])
        gammanominator[n, k] = pi[k]*gauss
        likelihood += pi[k]*gauss 
    ## when k loop finished, likelihood is the k sum    
    gamma[n,:] = gammanominator[n,:]/likelihood
    loglikelihood += np.log(likelihood)
        
plt.scatter(X, Y)
plt.show()
