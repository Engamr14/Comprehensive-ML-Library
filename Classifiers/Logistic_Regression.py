import numpy as np
import scipy.optimize as optimize
import scipy.linalg as sc 

class Logistic_Regression:
    def __init__(self, reg_param = 0.000001, pi = 0.5):
        self.reg_param = reg_param
        self.pi = pi
          
    def compute_loss(self, X, Y):    
        def j(v):   
            n_1 = (Y == 1).sum()
            n_0 = (Y == 0).sum()
            reg_term = (self.reg_param / 2) * np.linalg.norm(v[0:-1]) ** 2
            term_1 = -1 * np.dot(v[0:-1].T,X[:, Y == 1]) + v[-1]
            term_0 = np.dot(v[0:-1].T,X[:,Y == 0]) + v[-1]
            class_1 = np.logaddexp(0, term_1).sum() / n_1
            class_0 = np.logaddexp(0, term_0).sum() / n_0
            return reg_term + class_1*(self.pi) + class_0*(1 - self.pi)
        return j
        
    def fit(self, X, Y):
        loss = self.compute_loss(X, Y)
        x = optimize.fmin_l_bfgs_b(loss,np.zeros(X.shape[0]+1),approx_grad = True)
        self.w=x[0][0:X.shape[0]]
        self.b=x[0][-1]
            
    def predict(self, X):
        Z = np.dot(self.w.T,X)+self.b
        Y = 1 / np.logaddexp(0,-Z)
        Y = np.where(Y > 0.5, 1, 0)
        return Y
    
    def compute_scores(self, X):
        scores = np.dot(self.w.T, X) + self.b  # - np.log(self.nT/self.nF)
        return scores
    
        
        
        
        
        

    

        
        
    
        
        
        
        