import numpy as np
import scipy.optimize as optimize
import scipy.linalg as sc 

class Quad_Logistic_Regression:
    def __init__(self, reg_param = 0.0018, pi = 0.5):
        self.reg_param = reg_param
        self.pi = pi
        
    
    def compute_logistic_loss_qua_with_gard(self, X, Y):   
        def j(v):       
            x=v[0:-1]
            
            n_1 = (Y == 1).sum()
            n_0 = (Y == 0).sum()
            
            reg_term = (self.reg_param / 2) * np.linalg.norm(v[0:-1]) ** 2
            
            false_stacked=[np.hstack((np.dot(i.reshape((i.size,1)),i.reshape((1,i.size))).flatten('F'),i)).T for i in X[:,Y==0].T]
            true_stacked=[np.hstack((np.dot(i.reshape((i.size,1)),i.reshape((1,i.size))).flatten('F'),i)).T for i in X[:,Y==1].T]
     
            score_true=np.array([np.dot(x,i)+v[-1] for i in true_stacked]) 
            score_false=np.array([np.dot(x,i)+v[-1] for i in false_stacked]) 
            
            exp_true=np.exp(np.array(score_true)*(-1))
            exp_false=np.exp(np.array(score_false)*(1))
            
            b_grad_true=exp_true*(-1)/(1+exp_true)
            b_grad_false=exp_false*1/(1+exp_false)
            
            grad_true= self.pi*np.array([i*b_grad_true[idx] for idx,i in enumerate(true_stacked)])
            grad_false= (1-self.pi)*np.array([i*b_grad_false[idx] for idx,i in enumerate(false_stacked)])
            
            b_tot=(self.pi*b_grad_true.sum()/n_1+ (1-self.pi)*b_grad_false.sum()/n_0)
            tot=np.hstack(((self.reg_param*x+grad_true.sum(axis=0)/n_1+grad_false.sum(axis=0)/n_0),b_tot))
                       
            class_true=np.log(1+exp_true).sum()/n_1
            class_false=np.log(1+exp_false).sum()/n_0
                
            #a=v[:dtr.shape[0]**2]
            return reg_term +class_true*(self.pi) +class_false*(1-self.pi), tot     
        return j
    
    def fit(self, X, Y):
        self.w = None
        self.b = None
        loss = self.compute_logistic_loss_qua_with_gard(X, Y)
        x = optimize.fmin_l_bfgs_b(loss,np.zeros((X.shape[0]+1)*X.shape[0]+1))
        self.w = x[0][X.shape[0]**2:-1]
        self.b = x[0][-1]
        self.A = x[0][0:X.shape[0]**2].reshape((X.shape[0],X.shape[0]),order='F')        
    
    def predict(self, X):
        term = np.array([np.dot(x,np.dot(self.A,x.T)) for x in X.T])
        Z = np.dot(self.w.T,X) + self.b + term
        Y = 1 / np.logaddexp(0,-Z)
        Y = np.where(Y > 0.5, 1, 0)
        return Y

    def compute_scores(self, X):
        score = np.dot(self.w.T, X) + self.b  # - np.log(self.nT/self.nF)
        return score    
        
        
        
        
        
