import numpy as np
import scipy.optimize as optimize
import scipy.linalg as sc 

class Kernel_Support_Vector_Machine:
    def __init__(self, kernel, kernel_input_1, kernel_input_2 = None, K = 1, C = 0.1, fac = 10000000, pi = 0.5):
        self.K = K
        self.C = C
        self.fac = fac
        self.pi = pi
        if kernel == 'poly':
            self.kernel = self.get_kernal_poly(kernel_input_1, kernel_input_2)
        elif kernel == 'rad':
           self. kernel = self.get_kernal_rad(kernel_input_1)
        elif kernel == 'rad_weighted':
            self.kernel = self.get_kernal_rad_weighted(kernel_input_1, kernel_input_2)
        
    def get_kernal_poly(self, d, c, psi=0):
        def kernel(x,y):
            return (np.dot(x.T,y) + c) ** d + psi
        return kernel
    
    def get_kernal_rad(self, sig, psi=0):
        def kernel(x,y):
            return np.exp(-sig * np.linalg.norm(x-y) ** 2) + psi
        return kernel
    
    def get_kernal_rad_weighted(self, sig,w,psi=0):
        def kernel(x,y):
            return np.exp(-sig * np.linalg.norm((x - y) / w) ** 2) + psi
        return kernel

    def comp_loss_kernel_SVM(self, X, Y):
        b = Y * 2 - 1
        H=np.zeros((X.shape[1],X.shape[1]))
        for idx,i in enumerate(H):
            for idy ,_ in enumerate(i):
                H[idx,idy]=b[idx]*b[idy]*self.kernel(X[:,idx],X[:,idy])
        
        def j(alpha):
            loss=0.5*np.dot(alpha.T,np.dot(H,alpha))-np.dot(alpha.T,np.ones(alpha.shape))
            return loss,np.dot(H,alpha)-1
        return j

    def fit(self, X, Y):
        self.X_TR = X
        self.Y_TR = Y
        loss = self.comp_loss_kernel_SVM(X, Y)
        freq_true=(Y==1).sum()/Y.size
        freq_false=(Y==0).sum()/Y.size
        
        x = optimize.fmin_l_bfgs_b(loss,np.zeros(X.shape[1]),factr= self.fac,
                           bounds=[(0, self.C * self.pi/freq_true) if Y[i]==1 else (0, self.C *(1- self.pi)/freq_false) for i in range(X.shape[1])])
        self.alpha = x[0]
        

    def predict(self, X):
        b = self.Y_TR * 2 - 1
        z = self.alpha * b
        approx = []
        for i in X.T:
            approx.append((np.array([self.kernel(j,i) for j in self.X_TR.T]) * z).sum())
        
        approx = np.array(approx, dtype=np.float32)
        results = np.where(approx > 0, 1, 0)
        return results
    
    def compute_scores(self, X):
        b = self.Y_TR * 2 - 1
        z = self.alpha * b
        scores = []
        for i in X.T:
            scores.append((np.array([self.kernel(j,i) for j in self.X_TR.T]) * z).sum())
        return scores
    

    