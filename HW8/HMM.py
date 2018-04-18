import numpy as np
import matplotlib.pyplot as plt


def ReadData():
    Data = np.loadtxt('points.dat')
    DataX = Data[:,0]
    DataY = Data[:,1]
    cutoff = int(0.9*len(DataX))
    X_train = DataX[0:cutoff]
    Y_train = DataY[0:cutoff]
    X_dev = DataX[cutoff:]
    Y_dev = DataY[cutoff:]
    Xtra = np.zeros((len(X_train), 2),  dtype='float64')
    Xdev = np.zeros((len(X_dev), 2),  dtype='float64')
    for i in range(len(X_train)):
        Xtra[i,:] = np.array([X_train[i],Y_train[i]])
    for i in range(len(X_dev)):
        Xdev[i,:] = np.array([X_dev[i],X_dev[i]])
    return Xtra, Xdev

def TwoDGaussian(xinput, mean, cov):
    detcov = np.linalg.det(cov)
    temp = np.dot(np.transpose(xinput - mean), np.linalg.inv(cov))
    Epart = np.exp(-0.5*np.dot(temp, (xinput - mean)))
    return 1/(2*np.pi*np.sqrt(detcov))*Epart

def Initialize(Xtra, K):
    klen = K
    pi = []
    mu = np.zeros((klen, 2), dtype='float64')
    np.random.seed(367243221)
    for k in range(klen):
        pi.append(1/klen)
        #mu[k] = Xtra[k*100]
    mu = np.random.rand(klen,2)
    pi = np.asarray(pi,dtype='float64')
    sigma = np.zeros((klen,2,2),  dtype='float64')
    sigma[:,0,0] = 1.
    sigma[:,1,1] = 1.
    TransitionA = np.zeros((klen, klen), dtype='float64')+1./klen
    return pi, mu, sigma, TransitionA
 
def forward_backward(Xtra, pi, mu, sigma, TransitionA):
    nlen = len(Xtra[:,0])
    klen = len(pi)
    alpha = np.zeros((nlen, klen),  dtype='float64')
    scale = np.zeros(nlen,  dtype='float64')
    for k in range (klen):
        gauss = TwoDGaussian(Xtra[0,:], mu[k,:], sigma[k,:,:])
        alpha[0, k] = pi[k]*gauss
    ##scale[0] = 1
    scale[0] = np.sum(alpha[0, :])
    alpha[0, :] = alpha[0, :] / scale[0]
        
    for n in range (1, nlen):
        for k in range (klen): 
            gauss = TwoDGaussian(Xtra[n,:], mu[k,:], sigma[k,:,:])
            alpha[n, k] = gauss*sum([alpha[n-1, sumk]*TransitionA[sumk, k] for sumk in range(klen)])
        scale[n] = np.sum(alpha[n,:])/np.sum(alpha[n-1,:])
        alpha[n, :] = alpha[n, :]/ scale[n]
    
    
    beta = np.ones((nlen, klen),  dtype='float64')
    for n in range(nlen-2, -1, -1):
        for k in range(klen):
            gauss = TwoDGaussian(Xtra[n+1,:], mu[k,:], sigma[k,:,:])
            beta[n, k] = sum([beta[n+1, sumk]*TwoDGaussian(Xtra[n+1,:], mu[sumk,:], sigma[sumk,:,:])*
                               TransitionA[k, sumk] for sumk in range(klen)])
        beta[n, :] =  beta[n, :] / scale[n+1]
    return alpha, beta, scale

def EM(Xtra, pi, mu, sigma, TransitionA):
    nlen = len(Xtra[:,0])
    klen = len(pi)
    results = []    
    for interations in range (30):
        alpha, beta, scale = forward_backward(Xtra, pi, mu, sigma, TransitionA)
        
        ksi = np.zeros((nlen, klen, klen),  dtype='float64')
        gamma = np.ones((nlen, klen),  dtype='float64')
        gamma =  alpha * beta
    
        for n in range(1, nlen):
            for i in range (klen) :
                gauss = TwoDGaussian(Xtra[n,:], mu[i,:], sigma[i,:,:])
                for j in range (klen) :
                    ksi [n, i, j] = 1/scale[n]*alpha[n-1, j]*gauss*TransitionA[j, i]*beta[n, i]
    
        pi = gamma[0,:]/sum(gamma[0,:])
        
        denominator = np.sum(np.sum(ksi, axis = 0), axis = 0 )
        for j in range (klen):
            for k in range (klen):
                TransitionA[j, k] = sum(ksi [:, j, k])/denominator[j]
    
        for k in range (0, klen):
            tempnominator = np.array([0., 0.])
            tempdenominator = 0.
            for n in range (0, nlen):
                tempnominator += gamma[n,k]*Xtra[n,:]
                tempdenominator += gamma[n,k]
            mu[k,:] = tempnominator/tempdenominator
        
        for k in range (0, klen):
            tempnominator = np.array([[0., 0.], [0., 0.]])
            tempdenominator = 0.
            for n in range (0, nlen):
                tempnominator += gamma[n, k]*np.outer(Xtra[n,:]- mu[k], np.transpose(Xtra[n,:]- mu[k]))
                tempdenominator += gamma[n,k]
            sigma[k,:,:] = tempnominator/tempdenominator
        results.append(np.sum(np.log(scale))) 
    return results[1:]

def main():
    Xtra, Xdev = ReadData()
    
    pi, mu, sigma, TransitionA = Initialize(Xtra, 3)
    Ydata3 = EM(Xtra, pi, mu, sigma, TransitionA)

    pi, mu, sigma, TransitionA = Initialize(Xtra, 4)
    Ydata4 = EM(Xtra, pi, mu, sigma, TransitionA)

    pi, mu, sigma, TransitionA = Initialize(Xtra, 5)
    Ydata5 = EM(Xtra, pi, mu, sigma, TransitionA)

    pi, mu, sigma, TransitionA = Initialize(Xtra, 6)
    Ydata6 = EM(Xtra, pi, mu, sigma, TransitionA)

    plt.plot(Ydata3, label = '3')
    plt.plot(Ydata4, label = '4')
    plt.plot(Ydata5, label = '5')
    plt.plot(Ydata6, label = '6')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()










