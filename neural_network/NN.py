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
import keras.losses
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


feature_names = ['FTSEmedrvd', 'FTSErsd', 'FTSErs5ssd', 'GDAXImedrvd', 'GDAXIrs5ssd',
                 'RUTmedrvd', 'RUTrs5ssd', 'RUTrs5ssw', 'RUTrs5ssm', 'DJIrv10d',
                 'DJIrsd', 'DJIrs5ssd', 'DJIrs5ssw', 'DJIrs5ssm', 'IXICbv5ssd',
                 'IXICmedrvd', 'IXICrs5ssd', 'IXICrs5ssw', 'IXICrs5ssm', 'AEXrs5ssd',
                 'IBEXmedrvd', 'IBEXmedrvw', 'IBEXrs5ssd', 'IBEXrs5ssw', 'IBEXrs5ssm',
                 'STOXX50Erv10d', 'STOXX50Emedrvd', 'STOXX50Ers5ssd', 'STOXX50Ers5ssw',
                 'STOXX50Ers5ssm', 'FTSEMIBmedrvd', 'FTSEMIBrs5ssd', 'FTSEMIBrs5ssw',
                 'FTSEMIBrs5ssm']

# sklearn's scaler--standardization
sc = StandardScaler()
scaled_train_features = sc.fit_transform(train_features)
scaled_test_features = sc.fit_transform(test_features)

modelA = Sequential()
modelA.add(
    Dense(50, input_dim=scaled_train_features.shape[1], activation="relu"))
modelA.add(Dense(10, activation="relu"))
modelA.add(Dense(1, activation="linear"))

modelA.compile(optimizer='adam', loss='mse')
historyA = modelA.fit(scaled_train_features, train_targets, epochs=100)

# Show the plot -- loss function
plt.plot(historyA.history['loss'])
plt.title('Loss:'+str(round(historyA.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('M:/516/loss function.png')
plt.close()


# Neural network --- Model 1
# Calculate R^2 score
test_predsA = modelA.predict(scaled_test_features)
train_predsA = modelA.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_predsA))
print('Test R^2 score:', r2_score(test_targets, test_predsA))

# Show the plot -- regression
plt.scatter(train_predsA, train_targets, label='Train')
plt.scatter(test_predsA, test_targets, label='Test')
x = np.linspace(-1, 23, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/NN-model 1.png')
plt.close()


# Neural network --- Model 2
# Create loss function
def mean_squared_error(y_true, y_pred):
    loss = tf.square(y_true-y_pred)
    return tf.reduce_mean(loss, axis=-1)


# Enable use of loss with keras
keras.losses.mean_squared_error = mean_squared_error

# fit the model with our mse loss function
modelA.compile(optimizer='adam', loss=mean_squared_error)
historyB = modelA.fit(scaled_train_features, train_targets, epochs=100)

# Show the plot -- loss function
plt.plot(historyB.history['loss'])
plt.title('Loss:' + str(round(historyB.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Checking out performance
# Calculate R^2 score
test_predsB = modelA.predict(scaled_test_features)
train_predsB = modelA.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_predsB))
print('Test R^2 score:', r2_score(test_targets, test_predsB))

# Show the plot -- regression
plt.scatter(train_predsB, train_targets, label='Train')
plt.scatter(test_predsB, test_targets, label='Test')
x = np.linspace(-1, 28, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/NN-model 2.png')
plt.close()


# Neural network --- Model 3
# create loss function
def sign_penalty(y_true, y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true*y_pred, 0),
                    penalty*tf.square(y_true-y_pred),
                    tf.square(y_true-y_pred))
    return tf.reduce_mean(loss, axis=-1)


# Enable use of loss with keras
keras.losses.sign_penalty = sign_penalty

# Using the custom loss
# create the model
modelC = Sequential()
modelC.add(
    Dense(50, input_dim=scaled_train_features.shape[1], activation='relu'))
modelC.add(Dense(10, activation='relu'))
modelC.add(Dense(1, activation='linear'))

# fit the model with our costom 'sign_penalty' loss function
modelC.compile(optimizer='adam', loss=sign_penalty)
historyC = modelC.fit(scaled_train_features, train_targets, epochs=100)

# Show the plot -- loss function
plt.plot(historyC.history['loss'])
plt.title('Loss:'+str(round(historyC.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Checking out performance
# Calculate R^2 score
test_predsC = modelC.predict(scaled_test_features)
train_predsC = modelC.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_predsC))
print('Test R^2 score:', r2_score(test_targets, test_predsC))

# Show the plot -- regression
plt.scatter(train_predsC, train_targets, label='Train')
plt.scatter(test_predsC, test_targets, label='Test')
x = np.linspace(-0.5, 24, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/NN-model 3.png')
plt.close()


# Neural network --- Model 4
# No drop out
model_1 = Sequential()
model_1.add(
    Dense(50, input_dim=scaled_train_features.shape[1], activation='relu'))
model_1.add(Dense(10, activation='relu'))
model_1.add(Dense(1, activation='linear'))

# fit the model with our costom 'sign_penalty' loss function
model_1.compile(optimizer='adam', loss=sign_penalty)
history_1 = model_1.fit(scaled_train_features,
                        train_targets,
                        epochs=100)

# Show the plot -- loss function
plt.plot(history_1.history['loss'])
plt.title('Loss:'+str(round(history_1.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')

# calculate R^2 score
test_preds_1 = model_1.predict(scaled_test_features)
train_preds_1 = model_1.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_preds_1))
print('Test R^2 score:', r2_score(test_targets, test_preds_1))

# Show the plot -- regression
plt.scatter(train_preds_1, train_targets, label='Train')
plt.scatter(test_preds_1, test_targets, label='Test')
x = np.linspace(-0.5, 23, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/NN-model 4.png')
plt.close()


# Neural network --- Model 5
# Drop out in keras
model_2 = Sequential()
model_2.add(Dense(500,
                  input_dim=scaled_train_features.shape[1],
                  activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(1, activation='linear'))

# Fit the model with our costom 'sign_penalty' loss function
model_2.compile(optimizer='adam', loss=sign_penalty)
history_2 = model_2.fit(scaled_train_features,
                        train_targets,
                        epochs=100)

# Show the plot -- loss function
plt.plot(history_2.history['loss'])
plt.title('Loss:'+str(round(history_2.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Calculate R^2 score
test_preds_2 = model_2.predict(scaled_test_features)
train_preds_2 = model_2.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_preds_2))
print('Test R^2 score:', r2_score(test_targets, test_preds_2))

# Show the plot -- regression
plt.scatter(train_preds_2, train_targets, label='Train')
plt.scatter(test_preds_2, test_targets, label='Test')
x = np.linspace(-0.5, 22, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/NN-model 5.png')
plt.close()


# Neural network --- Model 6
# Implementing ensembling
# Make predictions from 2 neiral net models
train_pred1 = model_1.predict(scaled_train_features)
train_pred2 = model_2.predict(scaled_train_features)
test_pred1 = model_1.predict(scaled_test_features)
test_pred2 = model_2.predict(scaled_test_features)
# Horizontally stack predictions and take the average rows
train_preds_avg = np.mean(np.hstack((train_pred1, train_pred2)), axis=1)
test_preds_avg = np.mean(np.hstack((test_pred1, test_pred2)), axis=1)
print('Train R^2 score:', r2_score(train_targets, train_preds_avg))
print('Test R^2 score:', r2_score(test_targets, test_preds_avg))

# Show the plot -- regression
plt.scatter(train_preds_avg, train_targets, label='Train')
plt.scatter(test_preds_avg, test_targets, label='Test')
x = np.linspace(-0.5, 22, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/NN-model 6.png')
plt.close()
