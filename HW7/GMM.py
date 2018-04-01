#!/usr/bin/python3
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

def GMM(clusters, X, Y, X_dev, Y_dev, tiecontrol):
    klen = clusters
    nlen = len(X)
    Xave = np.mean(X)
    Yave = np.mean(Y)
    pi = []
    mu = np.zeros((klen, 2), dtype='float64')
    np.random.seed(36721321)
    noise = np.random.rand(klen,2)
    for k in range(klen):
        pi.append(1/klen)
        ##mu[k] = [Xave+noise[k, 0], Yave+noise[k, 1]]
        mu[k] = [X[k*50], Y[k*50]]
    pi = np.asarray(pi,dtype='float64')
    
    sigma = np.zeros((klen,2,2),  dtype='float64')
    sigma[:,0,0] = 1.
    sigma[:,1,1] = 1.
    
    gamma = np.zeros((len(X), len(pi)), dtype='float64')
    gammanominator = np.zeros((len(X), len(pi)), dtype='float64')
    
    
    results = []
    resultsdev = []
    nlendev = len(X_dev)
    
    for iterations in range(0, 60):
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

        loglikelihooddev = 0
        for n in range(0, nlendev):
            likelihooddev = 0
            xinput = np.array([X_dev[n],Y_dev[n]])
            for k in range(0, klen):
                gauss = TwoDGaussian(xinput, mu[k,:], sigma[k,:,:])
                likelihooddev += pi[k]*gauss    
            loglikelihooddev += np.log(likelihooddev)
   
        results.append(loglikelihood/9.0)
        resultsdev.append(loglikelihooddev)

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
            if (tiecontrol == 1) :
                temptie += sigma[k,:,:]*pi[k]
                for k in range(0, klen):
                    sigma[k,:,:] = temptie

    return results[1:], resultsdev[1:]

Data = np.loadtxt('points.dat')
DataX = Data[:,0]
DataY = Data[:,1]
cutoff = int(0.9*len(DataX))

X_train = DataX[0:cutoff]
Y_train = DataY[0:cutoff]
X_dev = DataX[cutoff:]
Y_dev = DataY[cutoff:]

#Ytrain3, Ydev3 = GMM(3, X_train, Y_train,X_dev, Y_dev, 0)
#Ytrain4, Ydev4 = GMM(4, X_train, Y_train,X_dev, Y_dev, 0)
#Ytrain5, Ydev5 = GMM(5, X_train, Y_train,X_dev, Y_dev, 0)
#Ytrain6, Ydev6 = GMM(6, X_train, Y_train,X_dev, Y_dev, 0)
#Ytrain7, Ydev7 = GMM(7, X_train, Y_train,X_dev, Y_dev, 0)

#xaixs = np.arange(len(Ytrain3))    
#plt.plot(xaixs,Ytrain3, label = '3')    
#plt.plot(xaixs,Ytrain4, label = '4')
#plt.plot(xaixs, Ytrain5, label = '5')
#plt.plot(xaixs, Ytrain6, label = '6')    
#plt.plot(xaixs, Ytrain7, label = '7')

#plt.plot(xaixs,Ydev3 , label = '3_dev')    
#plt.plot(xaixs,Ydev4 , label = '4_dev')
#plt.plot(xaixs,Ydev5 , label = '5_dev')    
#plt.plot(xaixs,Ydev6 , label = '6_dev')
#plt.plot(xaixs,Ydev7 , label = '7_dev')
#plt.legend()
#plt.show()

