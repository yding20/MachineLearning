import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def TwoDGaussian(xinput, mean, cov):
    detcov = np.linalg.det(cov)
    temp = np.dot(np.transpose(xinput - mean), np.linalg.inv(cov))
    Epart = np.exp(-0.5*np.dot(temp, (xinput - mean)))
    return 1/(2*np.pi*np.sqrt(detcov))*Epart

# Initilize the means mu, covariances sigma and mix coefficient pi
# and log likelihood, Test 4 clusters case.

def GMM(clusters, X, Y):
    klen = clusters
    nlen = len(X)
    Xave = np.mean(X)
    Yave = np.mean(Y)
    pi = []
    mu = np.zeros((klen, 2), dtype='float64')
    #np.random.seed(76321321)
    noise = np.random.rand(klen,2)-0.5
    for k in range(klen):
        pi.append(1/klen)
        mu[k] = [Xave+noise[k, 0], Yave+noise[k, 1]]
        ##mu[k] = [X[k*100], Y[k*100]]
    pi = np.asarray(pi,dtype='float64')
    
    sigma = np.zeros((klen,2,2),  dtype='float64')
    sigma[:,0,0] = 1.
    sigma[:,1,1] = 1.
    
    gamma = np.zeros((len(X), len(pi)), dtype='float64')
    gammanominator = np.zeros((len(X), len(pi)), dtype='float64')
    
    
    results = []
    
    for iterations in range(0, 40):
        loglikelihood = 0
        for n in range(0, nlen):
            likelihood = 0
            xinput = np.array([X[n],Y[n]])
            for k in range(0, klen):
                gauss = TwoDGaussian(xinput, mu[k,:], sigma[k,:,:])
                gammanominator[n, k] = pi[k]*gauss
                likelihood += pi[k]*gauss 
            ## when k loop finished, likelihood is the k sum    
            gamma[n,:] = gammanominator[n,:]/likelihood
            loglikelihood += np.log(likelihood)
            
        results.append(loglikelihood)
        effN = np.sum(gamma[:,:], axis=0)
        pi = effN / nlen
    
        for k in range(0, klen):
            tempmu = np.array([0., 0.])
            for n in range(0, nlen):
                xinput = np.array([X[n],Y[n]])
                tempmu += gamma[n,k]*xinput
            mu[k,:] = 1/effN[k]*tempmu
        
        temptie = 0
        for k in range(0, klen):
            tempsigma = np.array([[-0., 0.], [0., 0.]])
            for n in range(0, nlen):
                xinput = np.array([X[n],Y[n]])
                tempsigma += gamma[n, k]*np.outer(xinput- mu[k], np.transpose(xinput- mu[k]))
            sigma[k,:,:] = 1/effN[k]*tempsigma
            temptie += sigma[k,:,:]*pi[k]
        for k in range(0, klen):
            sigma[k,:,:] = temptie
    return results[1:]

Data = np.loadtxt('points.dat')
DataX = Data[:,0]
DataY = Data[:,1]
cutoff = int(0.9*len(DataX))

X_train = DataX[0:cutoff]
Y_train = DataY[0:cutoff]
X_dev = DataX[cutoff:]
Y_dev = DataY[cutoff:]

xaixs = np.arange(len(GMM(3, X_train, Y_train)))    
plt.plot(xaixs, GMM(3, X_train, Y_train), label = '3')    
plt.plot(xaixs, GMM(4, X_train, Y_train), label = '4')
plt.plot(xaixs, GMM(5, X_train, Y_train), label = '5')
plt.plot(xaixs, GMM(6, X_train, Y_train), label = '6')    
plt.plot(xaixs, GMM(7, X_train, Y_train), label = '7')
plt.legend()
plt.show()