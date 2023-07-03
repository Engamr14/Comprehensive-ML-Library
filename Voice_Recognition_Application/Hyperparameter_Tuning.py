import numpy as np
import matplotlib.pyplot as plt
######################################################################################
from Logistic_Regression import Logistic_Regression
from Quad_Logistic_Regression import Quad_Logistic_Regression
from Support_Vector_Machine import Support_Vector_Machine
from Kernel_Support_Vector_Machine import Kernel_Support_Vector_Machine
from Gaussian_Mixture_Model import Gaussian_Mixture_Model
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

############################ Start Tuning ############################################
print("\n############### Logistic Regression (Raw) ###################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('lr1')
    default_x_ticks = range(len(params))
    plt.title('Logistic Regression (Raw)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
 
print("\n############### Logistic Regression (PCA) ###################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_pca, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('lr2')
    default_x_ticks = range(len(params))
    plt.title('Logistic Regression (PCA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n############### Logistic Regression (LDA) ###################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('lr3')
    default_x_ticks = range(len(params))
    plt.title('Logistic Regression (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n############### Logistic Regression (Gaussianed) ###################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('lr4')
    default_x_ticks = range(len(params))
    plt.title('Logistic Regression (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))


print("\n############ Quad Logistic Regression (Raw) ################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Quad_Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('qlr1')
    default_x_ticks = range(len(params))
    plt.title('Quad Logistic Regression (Raw)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n############ Quad Logistic Regression (PCA) ################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Quad_Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_pca, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('qlr2')
    default_x_ticks = range(len(params))
    plt.title('Quad Logistic Regression (PCA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n############ Quad Logistic Regression (LDA) ################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Quad_Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('qlr3')
    default_x_ticks = range(len(params))
    plt.title('Quad Logistic Regression (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
    
print("\n############ Quad Logistic Regression (Gaussianed) ################")
params = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ticks = ['10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6']
results = []

for param in params:
    model = Quad_Logistic_Regression(reg_param = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('qlr4')
    default_x_ticks = range(len(params))
    plt.title('Quad Logistic Regression (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('Lamda')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########### Support Vector Machine (Raw) ###############")
params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ticks = ['10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3']
results = []
for param in params:
    model = Support_Vector_Machine(C = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X, Y, small_ratio = 0.2)
    results.append((scores, labels))
for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('svm1')
    default_x_ticks = range(len(params))
    plt.title('Support Vector Machine (Raw)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
   
print("\n########### Support Vector Machine (PCA) ###############")
params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ticks = ['10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3']
results = []

for param in params:
    model = Support_Vector_Machine(C = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_pca, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('svm2')
    default_x_ticks = range(len(params))
    plt.title('Support Vector Machine (PCA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
   
print("\n########### Support Vector Machine (LDA) ###############")
params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ticks = ['10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3']
results = []

for param in params:
    model = Support_Vector_Machine(C = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('svm3')
    default_x_ticks = range(len(params))
    plt.title('Support Vector Machine (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
  
print("\n########### Support Vector Machine (Gaussianed) ###############")
params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ticks = ['10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3']
results = []

for param in params:
    model = Support_Vector_Machine(C = param, pi = 0.5)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('svm4')
    default_x_ticks = range(len(params))
    plt.title('Support Vector Machine (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
    
print("\n########### Quad Support Vector Machine (Raw) ###############")
params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ticks = ['10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3']
results = []

for param in params:
    model = Kernel_Support_Vector_Machine('poly', 2, param)
    scores, labels = testing_tools.Single_Fold_Validation(model, X, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('qsvm1')
    default_x_ticks = range(len(params))
    plt.title('Quad Support Vector Machine (Raw)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########### Quad Support Vector Machine (LDA) ###############")
params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ticks = ['10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3']
results = []

for param in params:
    model = Kernel_Support_Vector_Machine('poly', 2, param)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('qsvm2')
    default_x_ticks = range(len(params))
    plt.title('Quad Support Vector Machine (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
    
print("\n########### Quad Support Vector Machine (Gaussianed) ###############")
params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ticks = ['10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3']
results = []

for param in params:
    model = Kernel_Support_Vector_Machine('poly', 2, param)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('qsvm3')
    default_x_ticks = range(len(params))
    plt.title('Quad Support Vector Machine (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########### RBF Support Vector Machine (Raw) ###############")
params = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ticks = ['0.2', '0.4', '0.6', '0.8', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
results = []

for param in params:
    model = Kernel_Support_Vector_Machine('rad', param)
    scores, labels = testing_tools.Single_Fold_Validation(model, X, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('rbfsvm1')
    default_x_ticks = range(len(params))
    plt.title('RBF Support Vector Machine (Raw)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########### RBF Support Vector Machine (LDA) ###############")
params = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ticks = ['0.2', '0.4', '0.6', '0.8', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
results = []

for param in params:
    model = Kernel_Support_Vector_Machine('rad', param)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('rbfqsvm2')
    default_x_ticks = range(len(params))
    plt.title('RBF Support Vector Machine (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########### RBF Support Vector Machine (Gaussianed) ###############")
params = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ticks = ['0.2', '0.4', '0.6', '0.8', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
results = []

for param in params:
    model = Kernel_Support_Vector_Machine('rad', param)
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('rbfqsvm3')
    default_x_ticks = range(len(params))
    plt.title('RBF Support Vector Machine (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########## Gaussian Mixture Model - Full Tied (LDA) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'full_tied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm3')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Full Tied (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
    
print("\n########## Gaussian Mixture Model - Full Tied (Gaussianed) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'full_tied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm4')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Full Tied (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########## Gaussian Mixture Model - Full Untied (LDA) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'full_untied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm3')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Full Untied (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
    
print("\n########## Gaussian Mixture Model - Full Untied (Gaussianed) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'full_untied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm4')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Full Untied (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########## Gaussian Mixture Model - Diagonal Untied (LDA) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'diag_untied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm3')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Diagonal Untied (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
    
print("\n########## Gaussian Mixture Model - Diagonal Untied (Gaussianed) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'diag_untied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm4')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Diagonal Untied (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

print("\n########## Gaussian Mixture Model - Diagonal Tied (LDA) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'diag_tied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_lda, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm3')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Diagonal Tied (LDA)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))
    
print("\n########## Gaussian Mixture Model - Diagonal Tied (Gaussianed) ##############")
params = [2, 3, 4, 5, 6, 7, 8, 9]
ticks = ['4', '8', '16', '32', '64', '128', '256', '512']
results = []

for param in params:
    model = Gaussian_Mixture_Model(n_components = param, mode = 'diag_tied')
    scores, labels = testing_tools.Single_Fold_Validation(model, X_gaussianed, Y, small_ratio = 0.2)
    results.append((scores, labels))

for pi in [0.1, 0.5, 0.9]:
    dcf_curve = []
    for result in results:
        min_dcf = testing_tools.compute_min_DCF(result[0], result[1], prior = pi)
        dcf_curve.append(min_dcf)
    plt.figure('gmm4')
    default_x_ticks = range(len(params))
    plt.title('Gaussian Mixture Model - Diagonal Tied (Gaussianed)')
    plt.plot(default_x_ticks, dcf_curve, label ='pi = ' + str(pi))
    plt.xticks(default_x_ticks, ticks)    
    plt.xlabel('number of components')
    plt.ylabel('DCF')
    plt.legend()
    min_dcf = min(dcf_curve)
    min_dcf_idx = dcf_curve.index(min_dcf)
    best_lamda = ticks[min_dcf_idx]
    print("pi = " + str(pi) +" >> best lamda is " + str(best_lamda))

