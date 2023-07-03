import numpy as np 
import math

def mcol(lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

class Gaussian_Model:
    def __init__(self, mode):
        self.mode = mode      
        
    def fit(self, X, Y):
        X0 = X[:,Y == 0]
        X1 = X[:,Y == 1]
        self.mu0 = mcol(X0.mean(1))
        self.mu1 = mcol(X1.mean(1))
        self.cov0, self.cov1 = self.compute_covariance(X0, X1, X.shape[1])           
            
    def predict(self, X):
        S = [[], []]
        SJoint = [[], []]
        SMarginal = [[], []]
        S[0] = self.Log_PDF(X,mcol(self.mu0),self.cov0)
        S[1] = self.Log_PDF(X,mcol(self.mu1),self.cov1)
        SJoint[0] = np.array(np.exp(S[0]), dtype=np.float32) / 2
        SJoint[1] = np.array(np.exp(S[1]), dtype=np.float32) / 2
        SMarginal[0] = mrow(SJoint[0].sum(axis = 0))
        SMarginal[1] = mrow(SJoint[1].sum(axis = 0))
        SPost = SJoint / SMarginal
        Labels = np.argmax(SPost, axis=0)
        return Labels
    
    def compute_scores(self, X):
        densities0 = self.Log_PDF(X, self.mu0, self.cov0)
        densities1 = self.Log_PDF(X, self.mu1, self.cov1)
        return densities1 - densities0
    
    def Log_PDF (self, X, mu, cov):
        con_inv = np.linalg.inv(cov)
        M = X.shape[0]
        const = -0.5 * M * np.log(2*np.pi)
        const += -0.5 * np.linalg.slogdet(cov)[1]    
        densities = []
        for i in range(X.shape[1]):
            x = X[:, i:i+1]
            density = const - 0.5 * np.dot((x-mu).T, np.dot(con_inv, (x-mu)))
            densities.append(density)    
        return np.array(densities).ravel()    

    def compute_covariance(self, X0, X1, length):
        Xc0 = X0 - self.mu0
        Xc1 = X1 - self.mu1
        nc0 = X0.shape[1]
        nc1 = X1.shape[1]
        cov0 = np.dot(Xc0, Xc0.T) / nc0
        cov1 = np.dot(Xc1, Xc1.T) / nc1
        if self.mode == 'full_tied':
            cov0 = (cov0 * nc0 + cov1 * nc1) / float(length)
            cov1 = (cov0 * nc0 + cov1 * nc1) / float(length)
        elif self.mode == 'diagonal_untied':
            cov0 = cov0 * np.eye(cov0.shape[0])
            cov1 = cov1 * np.eye(cov1.shape[0])
        elif self.mode == 'diagonal_tied':
            cov0 = cov0 * np.eye(cov0.shape[0])
            cov1 = cov1 * np.eye(cov1.shape[0])
            cov0 = (cov0 * nc0 + cov1 * nc1) / float(length)
            cov1 = (cov0 * nc0 + cov1 * nc1) / float(length)
        return cov0, cov1 
    
    
    
    
    
    
    
    
