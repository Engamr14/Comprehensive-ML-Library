import numpy as np
import matplotlib.pyplot as plt
#############################################################################################
from Logistic_Regression import Logistic_Regression
from Quad_Logistic_Regression import Quad_Logistic_Regression
from Support_Vector_Machine import Support_Vector_Machine
from Kernel_Support_Vector_Machine import Kernel_Support_Vector_Machine
from Gaussian_Mixture_Model import Gaussian_Mixture_Model
from Gaussian_Model import Gaussian_Model
from Data_Preprocessing import Data_Preprocessing
from Testing_Tools import Testing_Tools
############################## Instantiating Tools ##########################################
data_preprocessing = Data_Preprocessing()
testing_tools = Testing_Tools()
##################### Loading Datasets and Apply Preprocessing ##############################
X, Y = data_preprocessing.load_dataset('Train.txt')
X_pca, pca_mat = data_preprocessing.Dimensionality_Reduction_PCA(X, 4)
X_lda, lda_mat = data_preprocessing.Dimensionality_Reduction_LDA(X, Y, 4)
X_gaussianed = data_preprocessing.gausianize(X, X)
########################## Start Score Calibration ##########################################
threshold1 = -1 * np.log(0.1/(1 - 0.1))
threshold2 = -1 * np.log(0.5/(1 - 0.5))
threshold3 = -1 * np.log(0.9/(1 - 0.9))

models = [(Gaussian_Model(mode = 'full_tied'), threshold2, 0.5, 'raw', 'Gaussian Model / Full Tied / Raw'),
          (Logistic_Regression(reg_param = 0.000001, pi = 0.1), threshold2, 0.5, 'raw', 'Logistic Regression / Raw'),
          (Quad_Logistic_Regression(reg_param = 0.1, pi = 0.5), threshold2, 0.5, 'lda', 'Quad Logistic Regression / LDA'),
          (Support_Vector_Machine(C = 1), threshold2, 0.5, 'lda', 'Support Vector Machine / LDA'),
          (Kernel_Support_Vector_Machine('poly', 2, 0.01), threshold2, 0.5, 'lda', 'Quad Support Vector Machine / LDA'),
          (Kernel_Support_Vector_Machine('rad', 0.2), threshold2, 0.5, 'lda', 'RBF Support Vector Machine / LDA'),
          (Gaussian_Mixture_Model(4, 'full_untied'), threshold2, 0.5, 'lda', 'Gaussian Mixture Model / LDA')] 

# Drow Bayes Error Graph and get actual DCF
for model in models:
    if model[3] == 'raw':
        scores, labels = testing_tools.Kfold_Cross_Validation(model[0], X, Y)
    else:
        scores, labels = testing_tools.Kfold_Cross_Validation(model[0], X_lda, Y)
    testing_tools.draw_bayes_error(scores, labels)
    
    print('###### ' + model[4] + ' ######')
    for pi in [0.1, 0.5, 0.9]:        
        act_dcf = testing_tools.compute_act_DCF(scores, labels, model[1], pi)
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi)
        print('with prior = ' + str(pi) + ' -> minDCF = ' + str(min_dcf) + ', actDCF = ' + str(act_dcf))
  

