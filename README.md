### Content ###

## 1. Data ##
## 2. Traditional models ##
# 2.1 heterogeneous autoregressive (HAR) model #
## 3. Neural network models ##
# 3.1 Neural networks #
# 3.2 Recurrent neural network (RNN), long short-term memory network (LSTM) and gated recurrent unit (GRU) #
## 4. Supervised learning ##
# 4.1 Decision Trees (DT), Random Forests (RF) and Gradient Boosting Regression (GBR) # 
# 4.2 Support vector machine (SVM) #
## 5. Unsupervised learning ##
# 5.1 Principle component analysis (PCA) #


This folder contains three sub-folders containing MATLAB code for processing financial data. The data used in the code are various volatility estimations generated from intra-day returns (i.e. changes in stock prices measured within one trading day) from different national stock market indices from 2000 to 2016, which are available from the Oxford-MAN Realized Library (2021). Python code for traditional financial models, supervised machine learning models and unsupervised models are included which can be used in financial data analysis. 

## 1. Data ##
The Data folder includes the following files:
Data_Process.m is used for data processing, including generating lags for each variable. 
	loadData.py shows how to import the processed financial data, generate the covariance matrix and select data based on the required covariance threshold. 

## 2.Traditional models ##
The traditional folder includes the following files and models:  

# 2.1 heterogeneous autoregressive (HAR) model #
HAR.py shows how stock volatility data can be analysed using a heterogeneous autoregressive (HAR) model, which is the traditional forecasting model used for such data that does not involve machine learning. It forecasts stock volatility (realised variance) by considering the lag of the realised variance as well as the lag of the weekly and monthly moving average realised variance. 

## 3. Neural network models ##
The neural_network folder includes the following files and models:

# 3.1 Neural networks #
NN.py applies neural network models to the data. More specifically, a sequential model from the keras library is used with different loss functions. 
* Neural networks are models that take input data to be passed through several layers of neurons. In the hidden layers computations are performed on the data by multiplying the input by the weight of each neuron. The result at each neuron is passed through an activation function to see if the value computed is high enough to activate the neuron. Activated neurons pass data onto the next layer of neurons. Once it reaches the output layer, the loss of the network is calculated using the loss function. This is then used to adjust the weights of the neurons to reduce the loss. 

# 3.2 Recurrent neural network (RNN), long short-term memory network (LSTM) and gated recurrent unit (GRU) #
RNN_LSTM_GRU.py applies recurrent neural network (RNN), long short-term memory network (LSTM) and gated recurrent unit (GRU) models to the data. 

* Recurrent neural networks are neural networks designed to be good at processing sequential data for predictions. This is due to sequential memory. This is done through a looping mechanism in the neural network allowing previous information to be passed forward. 	

* LSTMs process data sequentially, passing on data as it propagates forward. LSTMs have mechanisms called cell states and gates. The cell state acts as a passage for the data to be passed down the sequence chain and acts as the memory of the network. The LSTM cell is comprised of 3 gates, the forget, input and output gates that either allow or disallow data to be added to the cell state. Gated cells allow information from previous LSTM or layer outputs to be stored in them, allowing them to act as the memory for the network. 

* GRUs are improved versions of the standard recurrent neural network. Similar to the LSTM, it looks to solve the vanishing gradient problem using gates.

## 4. Supervised learning ##
The supervised_learning folder includes the following files and models.

# 4.1 Decision Trees (DT), Random Forests (RF) and Gradient Boosting Regression (GBR) #
DF_RF_GBR.py imports the data and shows how it can be analysed with three unsupervised learning methods: Decision Trees (DT), Random Forests (RF) and Gradient Boosting Regression (GBR).
	
* Decision trees take the data and recursively split it on a binary tree until they are split down to leaf nodes. The best splits are found by maximising entropy gain on each split. Entropy is the measure of uncertainty in a group of observations, so by looking to maximise entropy, we look to narrow down the possible values at each split to as few as possible. 
		
* Random forests take the original data and creates new datasets through a technique called bootstrapping. This involves randomly sampling data from the original dataset and then putting it back. The bootstrapped datasets then 	individually have decision trees trained on a randomly selected subset of them. The predictions of all trees in the random forest are then combined and the mean is taken as the prediction.

* Gradient boosting regression uses a decision tree as a base estimator. It then recursively adds decision trees with one split to minimise loss (difference between predicted and actual value) at each stage of boosting. 

# 4.2 Support vector machine (SVM) #
SVM.py imports the data and shows how it can be analysed with support vector machines (SVM).

* Support vector regression uses support vector machines to predict continuous data by finding the best hyperplane that goes through as many datapoints as possible such that the margin around it (within the threshold value) covers all datapoints. 

## 5. Unsupervised learning ##
The unsupervised_learning folder includes PCA.py file. 

# 5.1 Principle component analysis (PCA) #
PCA.py file applies principle component analysis (PCA) to the data. 

* Principal component analysis is a dimensionality reduction method that looks to reduce the number of features in a dataset down to the most important ones while still retaining as much information as possible.

