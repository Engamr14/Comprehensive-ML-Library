import numpy as np
import matplotlib.pyplot as plt

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

class Testing_Tools:
    def __init__(self):
        pass
    
    def Test_Model(self, model, X_train, Y_train, X_test, Y_test):
        model.fit(X_train, Y_train)
        results = model.predict(X_test)
        #print(results)
        accuracy = self.Compute_Accuracy(Y_test, results)
        print("Testing Accuracy = ", round(accuracy, 1)," %")
        return accuracy
    
    def Kfold_Cross_Validation(self, model, X, Y, K = 5, pi = 0.5):
        n_samples = X.shape[1]
        X = X.T
        n_samples_per_fold = int(n_samples/K)
        starting_index = 0
        ending_index = n_samples_per_fold
        scores = []
        labels = []
        for i in range(K):
            # Compute testing samples
            X_test = X[starting_index : ending_index]
            Y_test = Y[starting_index : ending_index]
            labels.append(Y_test)
            
            # Compute training samples
            X_train_part1 = X[0 : starting_index]
            X_train_part2 = X[ending_index: -1]
            X_train = np.concatenate((X_train_part1, X_train_part2), axis = 0)
            
            Y_train_part1 = Y[0 : starting_index]
            Y_train_part2 = Y[ending_index: -1]
            Y_train = np.concatenate((Y_train_part1, Y_train_part2), axis = 0)
            
            # Apply to the model and get scores
            model.fit(X_train.T, Y_train)
            scores.append(model.compute_scores(X_test.T))
            
            # Updating indexes for next iteration
            starting_index += n_samples_per_fold
            ending_index += n_samples_per_fold                        
        return np.concatenate(scores), np.concatenate(labels) 
       
    def Single_Fold_Validation(self, model, X, Y, small_ratio, permutation_seed=0, fp_cost = 1, fn_cost = 1, pi = 0.5):
        # Split dataset to training & testing samples
        nTrain = int(X.shape[1] * (1 - small_ratio))
        np.random.seed(permutation_seed)
        idx = np.random.permutation(X.shape[1])
        idxTrain = idx[0:nTrain]
        idxTest = idx[nTrain:]
        X_train = X[:, idxTrain]
        X_test = X[:, idxTest]
        Y_train = Y[idxTrain]
        Y_test = Y[idxTest]        
        # Apply to the model and get scores
        model.fit(X_train, Y_train)
        scores = model.compute_scores(X_test)       
        return scores , Y_test 

    def Compute_Accuracy(self, Y, Y_predict):
        compare = Y_predict[Y_predict == Y]
        accuracy = compare.shape[0]/Y.shape[0] *100
        return accuracy   
    
    def compute_confusion_matrix(self, Pred, labels):
        conf_mat = np.zeros((2, 2))
        conf_mat[0, 0] = ((Pred == 0) * (labels == 0)).sum()
        conf_mat[0, 1] = ((Pred == 0) * (labels == 1)).sum()
        conf_mat[1, 0] = ((Pred == 1) * (labels == 0)).sum()
        conf_mat[1, 1] = ((Pred == 1) * (labels == 1)).sum()
        return conf_mat
    
    def compute_DCF(self, prior, fp_cost, fn_cost, conf_mat, norm=0):
        fpr = conf_mat[1,0] / (conf_mat[1,0] + conf_mat[0,0])
        fnr = conf_mat[0,1] / (conf_mat[0,1] + conf_mat[1,1])
        re= fpr * fp_cost * (1-prior) + fnr * fn_cost * prior
        if(norm == 0):
            re=re/min((1 - prior) * fp_cost, prior * fn_cost)
        return re    
    
    def compute_min_DCF(self, Y_pred, Y_label, prior = 0.5, fp_cost = 1, fn_cost = 1):
        global_dcf = 100
        for j in Y_pred:
            predicted = (Y_pred > j) * 1
            conf_mat = self.compute_confusion_matrix(predicted , Y_label)
            dcf = self.compute_DCF(prior, fp_cost, fn_cost, conf_mat)
            if(dcf < global_dcf):
                global_dcf = dcf
        return round(global_dcf, 3)

    def compute_act_DCF(self, scores, labels, threshold, prior = 0.5, fp_cost = 1, fn_cost = 1):
        conf_mat = self.compute_confusion_matrix(scores > threshold, labels)
        act_dcf = self.compute_DCF(prior, fp_cost, fn_cost, conf_mat)
        return round(act_dcf,3)
    
    def draw_bayes_error(self, score,label, c1='r',c2='b'):
        points=[]
        dcf_min=[]
        axis=np.linspace(-4,4,11)
        x_axis=[]
        for i in axis:
           
            p=1/(1+np.exp(-i))
            x=(p*1)/((1-p)*1)
            threshold=-np.log(x)
            x_axis.append(threshold)
            d = self.compute_DCF(p,1,1,self.compute_confusion_matrix(score > threshold, label))
            c = self.compute_min_DCF(score,label,p,1,1)
            dcf_min.append(c)
            points.append(d)

        plt.figure()
        plt.plot(x_axis,points, label='act DCF', color=c1)
        plt.plot(x_axis,dcf_min, label='mindcf', color=c2)
        plt.legend(loc='upper right')
        plt.ylim([0,1.1])
        plt.xlim([-4,4])
        plt.show()   
   
    
