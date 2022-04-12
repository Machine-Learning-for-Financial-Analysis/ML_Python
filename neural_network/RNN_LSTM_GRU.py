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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# 1. RNN model
# scale the data
sc = MinMaxScaler(feature_range=(0, 1))
scaled_train_features = sc.fit_transform(train_features)
scaled_test_features = sc.fit_transform(test_features)

# Reshaping X_train for efficient modelling
scaled_train_features = np.reshape(
    scaled_train_features, (scaled_train_features.shape[0], scaled_train_features.shape[1], 1))
scaled_test_features = np.reshape(
    scaled_test_features, (scaled_test_features.shape[0], scaled_test_features.shape[1], 1))
my_rnn_model = Sequential()
my_rnn_model.add(SimpleRNN(32, return_sequences=True))
my_rnn_model.add(SimpleRNN(32))
my_rnn_model.add(Dense(1))  # The time step of the output
my_rnn_model.compile(optimizer='rmsprop', loss='mean_squared_error')
historyA = my_rnn_model.fit(scaled_train_features,
                            train_targets, epochs=200, batch_size=150)

plt.plot(historyA.history['loss'])
plt.title('Loss:'+str(round(historyA.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('M:/516/RNN LOSS FUNCTION.png')
plt.close()

# calculate R^2 score
test_predsA = my_rnn_model.predict(scaled_test_features)
train_predsA = my_rnn_model.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_predsA))
print('Test R^2 score:', r2_score(test_targets, test_predsA))

plt.scatter(train_predsA, train_targets, label='Train')
plt.scatter(test_predsA, test_targets, label='Test')
x = np.linspace(-1, 15, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/RNN.png')
plt.close()

# 2. LSTM model
# The LSTM architecture
my_LSTM_model = Sequential()
my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(
    scaled_train_features.shape[1], 1), activation='tanh'))
#my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
#my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
my_LSTM_model.add(LSTM(units=50, activation='tanh'))
my_LSTM_model.add(Dropout(0.2))
my_LSTM_model.add(Dense(units=1))

# Compiling
my_LSTM_model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7,
                      momentum=0.9, nesterov=False), loss='mean_squared_error')
# Fitting to the training set
historyB = my_LSTM_model.fit(
    scaled_train_features, train_targets, epochs=200, batch_size=150)

plt.plot(historyB.history['loss'])
plt.title('Loss:'+str(round(historyB.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('M:/516/LSTM LOSS FUNCTION.png')
plt.close()

# calculate R^2 score
test_predsB = my_LSTM_model.predict(scaled_test_features)
train_predsB = my_LSTM_model.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_predsB))
print('Test R^2 score:', r2_score(test_targets, test_predsB))

plt.scatter(train_predsB, train_targets, label='Train')
plt.scatter(test_predsB, test_targets, label='Test')
x = np.linspace(-1, 20, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.ylim((-1, 50))
plt.savefig('M:/516/LSTM.png')
plt.close()

# 3. GRU model
# The GRU architecture
my_GRU_model = Sequential()
# First GRU layer with Dropout regularisation
my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(
    scaled_train_features.shape[1], 1), activation='tanh'))
my_GRU_model.add(Dropout(0.2))
# Second GRU layer
my_GRU_model.add(GRU(units=50, return_sequences=True, activation='tanh'))
my_GRU_model.add(Dropout(0.2))
# Third GRU layer
my_GRU_model.add(GRU(units=50, return_sequences=True, activation='tanh'))
my_GRU_model.add(Dropout(0.2))
# Fourth GRU layer
my_GRU_model.add(GRU(units=50, activation='tanh'))
my_GRU_model.add(Dropout(0.2))
# The output layer
my_GRU_model.add(Dense(units=1))

# Compiling the GRU
my_GRU_model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7,
                     momentum=0.9, nesterov=False), loss='mean_squared_error')
# Fitting to the training set
historyC = my_GRU_model.fit(scaled_train_features,
                            train_targets, epochs=100, batch_size=150)

plt.plot(historyC.history['loss'])
plt.title('Loss:'+str(round(historyC.history['loss'][-1], 6)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('M:/516/GRU LOSS FUNCTION.png')
plt.close()

# calculate R^2 score
test_predsC = my_GRU_model.predict(scaled_test_features)
train_predsC = my_GRU_model.predict(scaled_train_features)
print('Train R^2 score:', r2_score(train_targets, train_predsC))
print('Test R^2 score:', r2_score(test_targets, test_predsC))

plt.scatter(train_predsC, train_targets, label='Train')
plt.scatter(test_predsC, test_targets, label='Test')
x = np.linspace(-1, 25, 100)
plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
plt.legend(fontsize=20, loc=2)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.savefig('M:/516/GRU.png')
plt.close()
