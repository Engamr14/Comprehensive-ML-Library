import numpy as np
import scipy
import scipy.linalg as sc 
import scipy.stats
import scipy.optimize as op
import math
import matplotlib
import matplotlib.pyplot as plt


def mcol (lst):
    return lst.reshape((lst.shape[0],1))

def unique(list1): 
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)       
    return sorted(unique_list)


class Data_Preprocessing:
    def __init__(self):
        pass
    
    def load_dataset(self,File):
        Data = []
        Labels = []
        with open(File) as f:
            for line in f:
                try:
                        attrs = line.split(',')[0:-1]
                        attrs = mcol(np.array([float(i) for i in attrs]))
                        label = line.split(',')[-1]
                        Data.append(attrs)
                        Labels.append(label)
                except:
                    pass    
        Data = np.hstack(Data)
        Labels = np.array(Labels, dtype=np.int32)
        return Data, Labels
    
    def split_dataset (self,Data, Labels, small_ratio, permutation_seed=0):
        nTrain = int(Data.shape[1] * (1 - small_ratio))
        np.random.seed(permutation_seed)
        idx = np.random.permutation(Data.shape[1])
        idxTrain = idx[0:nTrain]
        idxTest = idx[nTrain:]
        D_TR = Data[:, idxTrain]
        D_TE = Data[:, idxTest]
        L_TR = Labels[idxTrain]
        L_TE = Labels[idxTest]
        return D_TR, L_TR, D_TE, L_TE
    
    def Dimensionality_Reduction_PCA (self, Data, m):
        mu = Data.mean(1)
        mu = mcol(mu)
    
        DataCentered = Data - mu
        Cov = np.dot(DataCentered,DataCentered.T)
        Cov = Cov / DataCentered.shape[1]
    
        s, U = np.linalg.eigh(Cov)
        P = U[:, ::-1][: , 0:m]
    
        DataProjected = np.dot(P.T,Data)
        return DataProjected, P
  
    
    def Dimensionality_Reduction_LDA (self, Data, labels, m):
        mu =Data.mean(1)
        mu= mcol(mu)
        D=[]
        clsMeans=[]
        n=[]
        SB = []
        clss = unique(labels)
        for i in clss:
            D.append(Data[:,labels == i])
            clsMeans.append(mcol(D[i].mean(1)))
            n.append(D[i].shape[1])
            SB.append(np.dot((clsMeans[i]-mu),(clsMeans[i]-mu).T))
        tot_SB = np.zeros((SB[0].shape[0],SB[0].shape[1]))
        for i in clss:
            SB[i] = SB[i]*n[i]
            tot_SB += SB[i]    
        tot_SB = tot_SB/Data.shape[1]
        DC=[]
        SW = []
        for i in clss:
            DC.append(D[i]-clsMeans[i])
            SW.append(np.dot(DC[i],DC[i].T))   
        tot_SW= sum(SW)
        tot_SW = tot_SW/Data.shape[1]
        s,U = scipy.linalg.eigh(tot_SB,tot_SW)
        W = U[:, ::-1][:,0:m]
        DataProjected = np.dot(W.T,Data)
        return DataProjected, W       
        
    def plot_bar(self, x_axis, y_axis, title):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(x_axis, y_axis)
        ax.set_title(title)
        plt.show()
        
    def plot_normal_distribution(self, Data, title):
        Mean = np.mean(Data, axis = 1)
        Var = np.var(Data, axis = 1)
        max_var = np.max(Var)
        mean_mean = np.mean(Mean)
        x = np.linspace(mean_mean - math.sqrt(max_var), mean_mean + math.sqrt(max_var), 100)
        for mean, var in zip(Mean, Var):
            plt.plot(x, scipy.stats.norm.pdf(x, mean , np.sqrt(var)))
        plt.title(title)
    
    
    def plot_normal_distribution_binary(self, mean0, mean1, var0, var1, title):
        max_var = np.max([var0, var1])
        max_mean = np.max([mean0, mean1])
        min_mean = np.min([mean0, mean1])
        x = np.linspace(min_mean - np.sqrt(max_var), max_mean + np.sqrt(max_var), 100)
        fig, ax = plt.subplots()
        ax.plot(x, scipy.stats.norm.pdf(x, mean0 , np.sqrt(var0)), label = 'Male')
        ax.plot(x, scipy.stats.norm.pdf(x, mean1 , np.sqrt(var1)), label = 'Female')
        ax.legend()
        ax.set_title(title)     
        
    def heatmap_pearson_correlation(self, Data, Cmap, fontColor, n_attributes, title):
        correlation_between_features = np.corrcoef(Data, Data) [0 : n_attributes, 0 : n_attributes]
        correlation_between_features = abs(correlation_between_features)
        
        fig, ax = plt.subplots()
        ax.imshow(correlation_between_features, cmap = Cmap, interpolation = 'nearest')
        ax.set_xticks(np.arange(n_attributes))
        ax.set_yticks(np.arange(n_attributes))
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(n_attributes):
            for j in range(n_attributes):
                ax.text(j, i, round(correlation_between_features[i, j], 1), ha="center", va="center", color = fontColor)
        ax.set_title(title)
        fig.tight_layout()
        plt.show()
        
    def gausianize(self, data_rank, data):
        v=np.zeros(data_rank.shape)
        for idx,i in  enumerate(data_rank):
            for idj,j in enumerate(i):
                v[idx,idj]=  scipy.stats.norm.ppf(((j<data[idx,:]).sum()+1)/(2+data.shape[1]))         
        return v
    
    def plot_scatter(self, Data, Labels, attributes):
        D0 = Data[:, Labels==0]
        D1 = Data[:, Labels==1]
    
        hFea = dict(zip(range(len(attributes)), attributes))
    
        for dIdx1 in range(Data.shape[0]):
            for dIdx2 in range(Data.shape[0]):
                if dIdx1 == dIdx2:
                    continue
                plt.figure()
                plt.xlabel(hFea[dIdx1])
                plt.ylabel(hFea[dIdx2])
                plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Male')
                plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Female')
            
                plt.legend()
                plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
                plt.savefig('../scatter_%d_%d.pdf' % (dIdx1, dIdx2))
            plt.show()
            
    def plot_hist(self, Data, Labels, attributes):
    
        D0 = Data[:, Labels==0]
        D1 = Data[:, Labels==1]
    
        hFea = dict(zip(range(len(attributes)), attributes))
    
        for feature_n in range(Data.shape[0]):
            plt.figure()
            plt.xlabel(hFea[feature_n])
            plt.hist(D0[feature_n, :], bins = 10, density = True, alpha = 0.4, label = 'Male')
            plt.hist(D1[feature_n, :], bins = 10, density = True, alpha = 0.4, label = 'Female')
            
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('../hist_%d.pdf' % feature_n)
        plt.show() 
        
    def plot_features(self, X, Y, attributes, label0, label1, name, defPath = ''):
        X0 = X[:, Y == 0]
        X1 = X[:, Y == 1]
        for i in range(X.shape[0]):
            fig = plt.figure()
            plt.title(attributes[i])
            plt.hist(X0[i, :], bins=70, density=True, alpha=0.7, facecolor='orange', label = label0, edgecolor='darkorange')
            plt.hist(X1[i, :], bins=70, density=True, alpha=0.7, facecolor='cornflowerblue', label = label1, edgecolor='royalblue')
            plt.legend(loc='best')
            plt.savefig(defPath + 'Dataset analysis charts/%s_%d.jpg' % (name, i), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
