import numpy as np
import matplotlib.pyplot as plt
######################################################################################
from Logistic_Regression import Logistic_Regression
from Quad_Logistic_Regression import Quad_Logistic_Regression
from Support_Vector_Machine import Support_Vector_Machine
from Kernel_Support_Vector_Machine import Kernel_Support_Vector_Machine
from Gaussian_Mixture_Model import Gaussian_Mixture_Model
from Gaussian_Model import Gaussian_Model
from Data_Preprocessing import Data_Preprocessing
from Testing_Tools import Testing_Tools
##################### Instantiating Tools and Models #################################
data_preprocessing = Data_Preprocessing()
testing_tools = Testing_Tools()

########################## Loading Datasets and Apply Preprocessing ##########################################
X, Y = data_preprocessing.load_dataset('Train.txt')
X_pca, pca_mat = data_preprocessing.Dimensionality_Reduction_PCA(X, 4)
X_lda, lda_mat = data_preprocessing.Dimensionality_Reduction_LDA(X, Y, 4)
X_gaussianed = data_preprocessing.gausianize(X, X)
X_gaussianed_pca, gaussian_pca_mat = data_preprocessing.Dimensionality_Reduction_PCA(X_gaussianed, 4)
X_gaussianed_lda, gaussian_lda_mat = data_preprocessing.Dimensionality_Reduction_LDA(X_gaussianed, Y, 4)
########################## Start Validation ##########################################

print("\n############## Gaussian Model - 1fold (Raw) ##################")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X, Y)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############## Gaussian Model - 5fold (Raw) ##################")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Single_Fold_Validation(model, X, Y, small_ratio = 0.2)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############## Gaussian Model - 1fold (PCA) ##################")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_pca, Y)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############## Gaussian Model - 5fold (PCA) ##################")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_pca, Y, small_ratio = 0.2)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############## Gaussian Model - 1fold (LDA) ##################")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############## Gaussian Model - 5fold (LDA) ##################")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)
    
print("\n########## Gaussian Model - 1fold (Gaussianed) ##############")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n########## Gaussian Model - 5fold (Gaussianed) ##############")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n########## Gaussian Model - 1fold (Gaussianed - PCA) ##############")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed_pca, Y)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n########## Gaussian Model - 5fold (Gaussianed - PCA) ##############")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed_pca, Y, small_ratio = 0.2)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)
    
print("\n########## Gaussian Model - 1fold (Gaussianed - LDA) ##############")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed_lda, Y)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n########## Gaussian Model - 5fold (Gaussianed - LDA) ##############")
print("mode  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for mode in ['full_untied', 'full_tied', 'diagonal_untied', 'diagonal_tied']:
    model = Gaussian_Model(mode)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed_lda, Y, small_ratio = 0.2)
    s = str(mode) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)
    
############################################################################################################
############################################################################################################   

print("\n############### Logistic Regression (Raw) ###################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Logistic_Regression(reg_param = 0.000001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############### Logistic Regression (PCA) ###################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Logistic_Regression(reg_param = 0.00001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_pca, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############### Logistic Regression (LDA) ###################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Logistic_Regression(reg_param = 0.00001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)
   
print("\n############### Logistic Regression (Gaussianed) ###################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Logistic_Regression(reg_param = 0.000001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############ Logistic Regression (Gaussianed - PCA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Logistic_Regression(reg_param = 0.00001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed_pca, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############ Logistic Regression (Gaussianed - LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Logistic_Regression(reg_param = 0.00001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed_lda, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############ Quad Logistic Regression (Raw) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Quad_Logistic_Regression(reg_param = 0.000001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)
    
print("\n############ Quad Logistic Regression (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Quad_Logistic_Regression(reg_param = 0.1, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)

print("\n############ Quad Logistic Regression (Gaussianed) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Quad_Logistic_Regression(reg_param = 0.0001, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)    
    
print("\n############ Support Vector Machine (Raw) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Support_Vector_Machine(C = 0.1, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s)  

print("\n############ Support Vector Machine (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Support_Vector_Machine(C = 1, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s) 

print("\n############ Support Vector Machine (Gausssianed) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
for pi1 in [0.1, 0.5, 0.9]:
    model = Support_Vector_Machine(C = 1, pi = pi1)
    scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
    s = str(pi1) + " |"
    for pi2 in [0.1, 0.5, 0.9]:
        min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
        s = s + "\t" + str(min_dcf)
    print(s) 

print("\n############ Quad Support Vector Machine (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")

model = Kernel_Support_Vector_Machine('poly', 2, kernel_input_2 = 0.01)
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)  

print("\n############ Quad Support Vector Machine (Gausssianed) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")

model = Kernel_Support_Vector_Machine('rad', 2, kernel_input_2 = 1)
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)     

print("\n############ RBF Support Vector Machine (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")

model = Kernel_Support_Vector_Machine('rad', 0.2)
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)  

print("\n############ RBF Support Vector Machine (Gausssianed) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
pi1 = 0.5
model = Kernel_Support_Vector_Machine('rad', 0.2)
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)

print("\n############ Gaussian Mixture Model - Full Tied (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
pi1 = 0.5
model = Gaussian_Mixture_Model(256)
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)  

print("\n######## Gaussian Mixture Model - Full Tied (Gausssianed) ############")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")

model = Gaussian_Mixture_Model(16)
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s) 

print("\n############ Gaussian Mixture Model - Full Untied (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
pi1 = 0.5
model = Gaussian_Mixture_Model(4, 'full')
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)  

print("\n######## Gaussian Mixture Model - Full Unied (Gausssianed) ############")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")

model = Gaussian_Mixture_Model(16, 'full')
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)

print("\n############ Gaussian Mixture Model - Diagonal Tied (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
pi1 = 0.5
model = Gaussian_Mixture_Model(512, 'diag_tied')
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)  

print("\n######## Gaussian Mixture Model - Diagonal Tied (Gausssianed) ############")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")

model = Gaussian_Mixture_Model(256, 'diag_tied')
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s) 

print("\n############ Gaussian Mixture Model - Diagonal Untied (LDA) ################")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")
pi1 = 0.5
model = Gaussian_Mixture_Model(16, 'diag_untied')
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_lda, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)  

print("\n######## Gaussian Mixture Model - Diagonal Unied (Gausssianed) ############")
print("pi  | \t0.1\t\t0.5\t\t0.9")
print("----------------------------")

model = Gaussian_Mixture_Model(256, 'diag_untied')
scores, labels = testing_tools.Kfold_Cross_Validation(model, X_gaussianed, Y)
s = str(pi1) + " |"
for pi2 in [0.1, 0.5, 0.9]:
    min_dcf = testing_tools.compute_min_DCF(scores, labels, pi2)
    s = s + "\t" + str(min_dcf)
print(s)
