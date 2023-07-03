import numpy as np
import scipy

def mcol (lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

def empirical_mean(X):
    return mcol(X.mean(1))

def empirical_covariance(X):
    mu = empirical_mean(X)
    C = np.dot((X - mu), (X - mu).T) / X.shape[1]
    return C

class Gaussian_Mixture_Model:
    def __init__(self, n_components, mode = 'full_untied', psi = 1e-2, alpha = 1e-1):
        self.n = n_components # 2^n components
        self.gmm0 = None
        self.gmm1 = None
        self.mode = mode
        self.psi = psi
        self.alpha = alpha
           
    def fit(self, D, L):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        self.gmm0 = self.GMM_LBG(D0)
        self.gmm1 = self.GMM_LBG(D1)

        return self

    def compute_scores(self, D):
        S, logD0 = self.logpdf_GMM(D, self.gmm0)
        S, logD1 = self.logpdf_GMM(D, self.gmm1)
        return logD1 - logD0
    
    def logpdf_GAU_ND(self, D, mu, sigma):
        P = np.linalg.inv(sigma)
        c1 = 0.5 * D.shape[0] * np.log(2 * np.pi)
        c2 = 0.5 * np.linalg.slogdet(P)[1]
        c3 = 0.5 * (np.dot(P, (D - mu)) * (D - mu)).sum(0)
        return - c1 + c2 - c3

    def logpdf_GMM(self, D, gmm):
        S = np.zeros((len(gmm), D.shape[1]))
        for g in range(len(gmm)):
            (w, mu, C) = gmm[g]
            S[g, :] = self.logpdf_GAU_ND(D, mu, C) + np.log(w)
        logD = scipy.special.logsumexp(S, axis=0)
        return S, logD
       
    def GMM_LBG(self,D):
        gmm = [(1, empirical_mean(D), empirical_covariance(D))]
        while len(gmm) <= self.n:
            gmm = self.GMM_EM(D, gmm)
            if(len(gmm) == self.n):
                break
            newGmm = []
            for i in range(len(gmm)):
                (w, mu, sigma) = gmm[i]
                U, s, Vh = np.linalg.svd(sigma)
                d = U[:, 0:1] * (s[0] ** 0.5) * self.alpha
                newGmm.append((w / 2, mu + d, sigma))
                newGmm.append((w / 2, mu - d, sigma))
            gmm = newGmm
        return gmm
    
    
    def GMM_EM(self,DT, gmm, diff = 1e-6):
        D, N = DT.shape
        to = None
        tn = None
    
        while to == None or tn - to > diff:
            to = tn
            S, logD = self.logpdf_GMM(DT, gmm)
            tn = logD.sum() / N
            P = np.exp(S - logD)
    
            newGmm = []
            sigmaTied = np.zeros((D, D))
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (mrow(gamma) * DT).sum(1)
                S = np.dot(DT, (mrow(gamma) * DT).T)
                w = Z/P.sum()
                mu = mcol(F / Z)
                sigma = (S / Z) - np.dot(mu, mu.T)
                if self.mode == 'full_tied':
                    sigmaTied += Z * sigma
                    newGmm.append((w, mu))
                    continue
                elif self.mode == 'diag_tied':
                    sigma *= np.eye(sigma.shape[0])
                    sigmaTied += Z * sigma
                    newGmm.append((w, mu))
                    continue
                elif self.mode == 'diag_untied':
                    sigma *= np.eye(sigma.shape[0])
                U, s, _ = np.linalg.svd(sigma)
                s[s<self.psi] = self.psi
                sigma = np.dot(U, mcol(s) * U.T)
                newGmm.append((w, mu, sigma))
    
            if self.mode == 'full_tied' or self.mode == 'diag_tied':
                sigmaTied /= N
                U, s, _ = np.linalg.svd(sigmaTied)
                s[s<self.psi] = self.psi
                sigmaTied = np.dot(U, mcol(s) * U.T)
                newGmm2 = []
                for i in range(len(newGmm)):
                    (w, mu) = newGmm[i]
                    newGmm2.append((w, mu, sigmaTied))
                newGmm = newGmm2
            gmm = newGmm
    
        return gmm
