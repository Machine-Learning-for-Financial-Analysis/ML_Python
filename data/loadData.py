import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the data from a certain path
df = pd.read_csv('All_variables_3rd_edition.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# 1. caculate the correlative Var and reduce high cov
# Create the correlation matrix
corr = df.corr().abs()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap='Blues', center=0,
            linewidths=1, annot=True, fmt=".2f")
plt.savefig('M:/516/corr.png')
plt.close()

# We can print how many parameters still retained under different thresholds,
# different thresholds cause different results, so we choose threshold = 0.95
threshold = 0.95
# Calculate the correlation matrix and take the absolute value
corr_matrix = df.corr().abs()
# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)
# List column names of highly correlated features (r > i)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
# Drop the features in the to_drop list
df_reduced = df.drop(to_drop, axis=1)
print("Dimensionality reduced from {} to {} columns using a correlation threshold of {}.".format(
    df.shape[1], df_reduced.shape[1], threshold))

# Use these code to show the correlation graph result after ruduced
corr = df_reduced.corr().abs()
mask = np.triu(np.ones_like(corr, dtype=bool))
#sns.heatmap(corr, mask=mask, cmap='Blues', center=0,
#            linewidths=1, annot=True, fmt=".2f")
corr = df_reduced.corr().abs()
mask = np.triu(np.ones_like(corr, dtype=bool))
#sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f")

# 2. Establish the module
features = df_reduced.drop(['RV'], axis=1)
targets = df['RV']
# Splitting the dataset into the Training set and Test set
# Fitting Simple Linear Regression to the Training set
train_features, test_features, train_targets,  test_targets = train_test_split(
    features, targets, test_size=1/3, random_state=42)
