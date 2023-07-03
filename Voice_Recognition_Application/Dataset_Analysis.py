import numpy as np
from Data_Preprocessing import Data_Preprocessing
import scipy.stats as stats
import math
import matplotlib.pyplot as plt

data_preprocessing = Data_Preprocessing()

Data, Labels = data_preprocessing.load_dataset("Train.txt")
Data_pca, pca_mat = data_preprocessing.Dimensionality_Reduction_PCA(Data, 4)
Data_lda, lda_mat = data_preprocessing.Dimensionality_Reduction_LDA(Data, Labels, 4)
Data_gaussianed = data_preprocessing.gausianize(Data, Data)

attributes = ['attr_1','attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6',
              'attr_7', 'attr_8', 'attr_9', 'attr_10', 'attr_11', 'attr_12']
reduced_attributes = ['attr_1','attr_2', 'attr_3', 'attr_4']

print('Data shape: ', Data.shape)
print('Labels shape: ', Labels.shape)

data_preprocessing.plot_features(Data, Labels, attributes, 'Male', 'Female', 'plot_raw_features')
data_preprocessing.plot_features(Data_gaussianed, Labels, attributes, 'Male', 'Female', 'plot_gaussianized_features')

data_preprocessing.plot_scatter(Data, Labels, attributes)
data_preprocessing.plot_scatter(Data_pca, Labels, reduced_attributes)
data_preprocessing.plot_scatter(Data_lda, Labels, reduced_attributes)



D0 = Data[:, Labels == 0]
D1 = Data[:, Labels == 1]
data_preprocessing.heatmap_pearson_correlation(Data, 'Greys', 'dimgray', 12, 'Correlation Between Raw Features (Total)')
data_preprocessing.heatmap_pearson_correlation(D0, 'Blues', 'k', 12, 'Correlation Between Raw Features (Male)')
data_preprocessing.heatmap_pearson_correlation(D1, 'Reds', 'k', 12, 'Correlation Between Raw Features (Female)')



'''
D0 = Data_pca[:, Labels == 0]
D1 = Data_pca[:, Labels == 1]
data_preprocessing.heatmap_pearson_correlation(Data, 'Greys', 'tab:brown', 4, 'Correlation Between PCA Reduced Features (Total)')
data_preprocessing.heatmap_pearson_correlation(D0, 'Blues', 'k', 4, 'Correlation Between PCA Reduced Features (Male)')
data_preprocessing.heatmap_pearson_correlation(D1, 'Reds', 'k', 4, 'Correlation Between PCA Reduced Features (Female)')

D0 = Data_lda[:, Labels == 0]
D1 = Data_lda[:, Labels == 1]
data_preprocessing.heatmap_pearson_correlation(Data, 'Greys', 'tab:brown', 4, 'Correlation Between LDA Reduced Features (Total)')
data_preprocessing.heatmap_pearson_correlation(D0, 'Blues', 'k', 4, 'Correlation Between LDA Reduced Features (Male)')
data_preprocessing.heatmap_pearson_correlation(D1, 'Reds', 'k', 4, 'Correlation Between LDA Reduced Features (Female)')

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

ax = sns.heatmap(correlation_between_features, linewidth=0.5)
plt.show()


mean0 = np.mean(Data[:, Labels == 0])
mean1 = np.mean(Data[:, Labels == 1])
var0 = np.var(Data[:, Labels == 0])
var1 = np.var(Data[:, Labels == 1])

data_preprocessing.plot_normal_distribution_binary(mean0, mean1, var0, var1, 'Normal Distribution of Data')

mean0 = np.mean(Data_pca[:, Labels == 0])
mean1 = np.mean(Data_pca[:, Labels == 1])
var0 = np.var(Data_pca[:, Labels == 0])
var1 = np.var(Data_pca[:, Labels == 1])

data_preprocessing.plot_normal_distribution_binary(mean0, mean1, var0, var1, 'Normal Distribution of Reduced Data (PCA)')

mean0 = np.mean(Data_lda[:, Labels == 0])
mean1 = np.mean(Data_lda[:, Labels == 1])
var0 = np.var(Data_lda[:, Labels == 0])
var1 = np.var(Data_lda[:, Labels == 1])

data_preprocessing.plot_normal_distribution_binary(mean0, mean1, var0, var1, 'Normal Distribution of Reduced Data (LDA)')


data_preprocessing.plot_bar(attributes, mean0, 'Male Mean')
data_preprocessing.plot_bar(attributes, mean1, 'Female Mean')

data_preprocessing.plot_bar(attributes, var0, 'Male Variance')
data_preprocessing.plot_bar(attributes, var1, 'Female Variance')

data_preprocessing.plot_hist(Data, Labels, attributes)

data_preprocessing.plot_scatter(Data, Labels, attributes)

data_preprocessing.plot_normal_distribution(Data, 'Normal distribution of data')

reduced_data=data_preprocessing.Dimensionality_Reduction_PCA(Data,8)
reduced_data=data_preprocessing.Dimensionality_Reduction_LDA(reduced_data,Labels,2)

data_preprocessing.plot_scatter(reduced_data, Labels, ['Dim0', 'Dim1'])
data_preprocessing.plot_normal_distribution(reduced_data, 'Normal distribution of reduced-dimension data')

mean0 = np.mean(reduced_data[:, Labels == 0], axis = 1)
mean1 = np.mean(reduced_data[:, Labels == 1], axis = 1)
var0 = np.var(reduced_data[:, Labels == 0], axis = 1)
var1 = np.var(reduced_data[:, Labels == 1], axis = 1)

data_preprocessing.plot_normal_distribution_binary(mean0, mean1, var0, var1)
'''






