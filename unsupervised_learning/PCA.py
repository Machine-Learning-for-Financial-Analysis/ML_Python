~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Machine leanring for financial analysis by Yanying Guan, Yanlun Zhu is licensed under 
# a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# Based on a work at https://realized.oxford-man.ox.ac.uk/.
# Copy this code to let your visitors know!
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Machine leanring for financial analysis</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/YanyingGuan; https://github.com/YanlunZhu" property="cc:attributionName" rel="cc:attributionURL">YanyingGuan, YanlunZhu</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://realized.oxford-man.ox.ac.uk/" rel="dct:source">https://realized.oxford-man.ox.ac.uk/</a>.
# Copyright (c) March 2022 Yanying Guan, Yanlun Zhu. All rights reserved.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the datasets
df = pd.read_csv('All.csv', dtype=float, delimiter=',')

# choose the features data and targets data
features = df.drop(['RV'], axis = 1)
targets = df['RV'] # 0 means number less than 1.22, 1 means number greater than 1.22, 1.22 is the 'RV' mean.

# Scale data before applying PCA
scaling = StandardScaler()
# Use fit and transform method
scaling.fit(features)
Scaled_data = scaling.transform(features)

# Select the number of components while preserving 95% of the variability in the data
pca = PCA(n_components = 0.95, random_state = 1)
pca.fit(Scaled_data)
data_new = pca.transform(Scaled_data)

# Print the var_ratio, var and n components
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
#print (pca.n_components_)

# Choose the right number of dimensions: When n_component about 8, the main component reach 95%
pca.explained_variance_ratio_ * 100
np.cumsum(pca.explained_variance_ratio_ * 100)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.savefig('elbow_plot.jpg', dpi = 100)

# Show the 2D plot
plt.figure(figsize = (10,10))
plt.scatter(data_new[:,0], data_new[:,1], data_new[:,2], c = targets, cmap='autumn')
#sns.scatterplot(data_new[:, 0], data_new[:, 1], marker = 'o', hue = df['RV'], palette = ['green', 'blue'])
plt.title("2D Scarttplot: 78.47% of the variability capture", pad = 15)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.grid()

# Show the 3D plot
# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (10,10))
# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection = '3d')

# data_new[:,0]is pc1,data_new[:,1] is pc2 while data_new[:,2] is pc3
axis.scatter(data_new[:,0], data_new[:,1], data_new[:,2], c = targets, cmap='plasma')
axis.set_title("3D Scarttplot: 84.11% of the variability capture", pad = 15)
axis.set_xlabel('First principle component', fontsize = 10)
axis.set_ylabel('Second principle component', fontsize = 10)
axis.set_zlabel('Third principle component', fontsize = 10)
plt.show()

# The data_new array holds the values of all 9 principle components
"""plt.figure(figsize = (10,10))
plt.plot(data_new)
plt.title("Transformed data by the principal components (95% Variability)", pad = 15)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.savefig('All principal.jpg', dpi = 1000)
# Create a pandas DataFrame using the values of all 9 principal components 
# and the label column of the original dataset
df_new = pd.DataFrame(data_new, columns = ['PC1', 'PC2', 'PC3','PC4','PC5', 'PC6', 'PC7', 'PC8', 'PC9'])
df_new['label'] = targets
print(df_new.head())
print(df_new.shape)"""
