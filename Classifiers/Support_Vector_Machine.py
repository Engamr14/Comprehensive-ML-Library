import numpy as np
import scipy.optimize as optimize
import scipy.linalg as sc 

class Support_Vector_Machine:
    def __init__(self, K = 1, C = 0.1, fac = 10000000, pi = 0.5):
        self.K = K
        self.C = C
        self.fac = fac
        self.pi = pi

    def compute_loss_linear_SVM(self, X,Y):
        b=Y*2-1
        z=X*b
        H=np.dot(z.T,z)
        def j(alpha):
            loss=0.5*np.dot(alpha.T,np.dot(H,alpha))-np.dot(alpha.T,np.ones(alpha.shape))
            return loss,np.dot(H,alpha)-1
        return j,z
    
    def fit(self, X, Y):
        X_prime=np.vstack((X,np.ones(X.shape[1])*self.K))
        loss, z = self.compute_loss_linear_SVM(X_prime,Y)
        freq_true=(Y==1).sum()/Y.size
        freq_false=(Y==0).sum()/Y.size
        bounds=[(0,self.C * self.pi / freq_true) if Y[i]==1 else (0,self.C*(1 - self.pi) / freq_false)
                for i in range(X_prime.shape[1])]
        x=optimize.fmin_l_bfgs_b(loss,np.zeros(X.shape[1]), factr = self.fac, bounds = bounds)       
        w_prime=(x[0]*z).sum(axis=1)
        self.w=w_prime[0:-1]
        self.b=w_prime[-1]*self.K
        
    def predict(self, X):
        approx = (np.dot(self.w.reshape((1,self.w.size)), X) + self.b)[0]
        results = np.where(approx > 0, 1, 0)
        return results
    
    def compute_scores(self, X):
        score = (np.dot(self.w.reshape((1, self.w.size)), X) + self.b)[0]
        return score
    

    