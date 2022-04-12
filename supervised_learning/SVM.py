~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Machine leanring for financial analysis by Yanying Guan, Yanlun Zhu is licensed under 
# a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# Based on a work at https://realized.oxford-man.ox.ac.uk/.
# Copy this code to let your visitors know!
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Machine leanring for financial analysis</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/YanyingGuan; https://github.com/YanlunZhu" property="cc:attributionName" rel="cc:attributionURL">YanyingGuan, YanlunZhu</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://realized.oxford-man.ox.ac.uk/" rel="dct:source">https://realized.oxford-man.ox.ac.uk/</a>.
# Copyright (c) March 2022 Yanying Guan, Yanlun Zhu. All rights reserved.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the datasets
df = pd.read_csv('All.csv', dtype=float, delimiter=',')

# remove features based on correlation coefficient
threshold = 0.95
# Calculate the correlation matrix and take the absolute value
corr_matrix = df.corr().abs()
# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
tri_df = corr_matrix.mask(mask)
# List column names of highly correlated features (r > i)
to_drop = [c for c in tri_df.columns if any(tri_df[c] >  threshold)]
# Drop the features in the to_drop list
df_reduced = df.drop(to_drop, axis = 1)
print("Dimensionality reduced from {} to {} columns using a correlation threshold of {}.".format(df.shape[1], df_reduced.shape[1], threshold))

# choose the features data and targets data
features = df_reduced.drop(['RV'], axis = 1)
targets = df['RV']

# Splitting the dataset into the Training set and Test set
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size = 0.8, random_state = 1)
classifier = svm.LinearSVR(random_state = 1)
classifier.fit(train_features, train_targets) 

# train score and test score
print("Train：", classifier.score(train_features,train_targets))
print("Test：", classifier.score(test_features,test_targets))

# show the plot
train_predictions = classifier.predict(train_features)
test_predictions = classifier.predict(test_features)
plt.scatter(train_predictions, train_targets, label='Train')
plt.scatter(test_predictions, test_targets, label='Test')
x = np.linspace(-1,28,100)
plt.plot(x, x, '-r', label = 'Actual=Prediction',linewidth = 1.5)
plt.axhline(y = 10, color = '#838B8B', linewidth = 1, linestyle = '--')
plt.legend(fontsize = 20, loc = 2)
plt.xlabel('Predictions',fontsize = 10)
plt.ylabel('Actual',fontsize = 10)
plt.show()
