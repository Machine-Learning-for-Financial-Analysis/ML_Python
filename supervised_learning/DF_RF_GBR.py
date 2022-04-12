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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# 1. Decision tree
def decisionTree():
    decision_tree = DecisionTreeRegressor(max_depth=5)
    decision_tree.fit(train_features, train_targets)

    # Print train score and test score
    print('Train score:', decision_tree.score(train_features, train_targets))
    print('Test score:', decision_tree.score(test_features, test_targets))

    # Calculate R^2 score
    train_predictions = decision_tree.predict(train_features)
    test_predictions = decision_tree.predict(test_features)
    print('Train R^2:', r2_score(train_targets, train_predictions))
    print('Test R^2:', r2_score(test_targets, test_predictions))

    # Show the plot
    train_predictions = decision_tree.predict(train_features)
    test_predictions = decision_tree.predict(test_features)
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.scatter(train_predictions, train_targets, label='Train')
    plt.scatter(test_predictions, test_targets, label='Test')
    x = np.linspace(-1, 70, 100)
    plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
    plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
    plt.legend(fontsize=20, loc=2)
    plt.xlabel('Predictions', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.savefig('M:/516/decisionTree.png')
    plt.close()


# 2. Random forest
# Print train score and test score
def randomForest():
    random_forestA = RandomForestRegressor()
    random_forestA.fit(train_features, train_targets)
    print('Train score:', random_forestA.score(train_features, train_targets))
    print('Test score:', random_forestA.score(test_features, test_targets))

    # Calculate R^2 score
    train_predictions_RFA = random_forestA.predict(train_features)
    test_predictions_RFA = random_forestA.predict(test_features)
    print('Train R^2:', r2_score(train_targets, train_predictions_RFA))
    print('Test R^2:', r2_score(test_targets, test_predictions_RFA))

    # Show the plot
    train_predictions_RFA = random_forestA.predict(train_features)
    test_predictions_RFA = random_forestA.predict(test_features)
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.scatter(train_predictions_RFA, train_targets, label='Train')
    plt.scatter(test_predictions_RFA, test_targets, label='Test')
    x = np.linspace(-0.5, 70, 100)
    plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
    plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
    plt.legend(fontsize=20, loc=2)
    plt.xlabel('Predictions', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.savefig('M:/516/randomForest.png')
    plt.close()


# 3. Feature importances and gradient boosting
def featureImp():
    random_forestC = RandomForestRegressor()
    random_forestC.fit(train_features, train_targets)
    # Feature importances from random forest model
    feature_importances = random_forestC.feature_importances_
    #print(feature_importances, train_features.columns)

    # Feature importances from random forest model
    importances = random_forestC.feature_importances_
    feature_names = ['FTSEmedrvd', 'FTSErsd', 'FTSErs5ssd', 'GDAXImedrvd', 'GDAXIrs5ssd',
                    'RUTmedrvd', 'RUTrs5ssd', 'RUTrs5ssw', 'RUTrs5ssm', 'DJIrv10d',
                    'DJIrsd', 'DJIrs5ssd', 'DJIrs5ssw', 'DJIrs5ssm', 'IXICbv5ssd',
                    'IXICmedrvd', 'IXICrs5ssd', 'IXICrs5ssw', 'IXICrs5ssm', 'AEXrs5ssd',
                    'IBEXmedrvd', 'IBEXmedrvw', 'IBEXrs5ssd', 'IBEXrs5ssw', 'IBEXrs5ssm',
                    'STOXX50Erv10d', 'STOXX50Emedrvd', 'STOXX50Ers5ssd', 'STOXX50Ers5ssw',
                    'STOXX50Ers5ssm', 'FTSEMIBmedrvd', 'FTSEMIBrs5ssd', 'FTSEMIBrs5ssw',
                    'FTSEMIBrs5ssm']

    # Index of greatest to least feature importances
    sorted_index = np.argsort(importances)[::-1]
    x = range(len(importances))
    # Create tick labels
    labels = np.array(feature_names)[sorted_index]
    plt.bar(x, importances[sorted_index], tick_label=labels)
    plt.axhline(y=0.01, color='#838B8B', linewidth=4, linestyle='--')
    # Rotate tock labels to vertical
    plt.xticks(rotation=90)
    plt.savefig('M:/516/featureImp.png')
    plt.close()


def gradientBoosting():
    gbr = GradientBoostingRegressor(learning_rate=0.01, n_estimators=200,
                                    subsample=0.6, random_state=42)

    gbr.fit(train_features, train_targets)
    train_scores = gbr.score(train_features, train_targets)
    test_scores = gbr.score(test_features, test_targets)
    print('Train score:', train_scores)
    print('Test score:', test_scores)

    # Calculate R^2 score
    train_predictions_GBR = gbr.predict(train_features)
    test_predictions_GBR = gbr.predict(test_features)
    print('Train R^2:', r2_score(train_targets, train_predictions_GBR))
    print('Test R^2:', r2_score(test_targets, test_predictions_GBR))

    # Show the plot
    train_predictions_GBR = gbr.predict(train_features)
    test_predictions_GBR = gbr.predict(test_features)
    plt.scatter(train_predictions_GBR, train_targets, label='Train')
    plt.scatter(test_predictions_GBR, test_targets, label='Test')
    x = np.linspace(-0.5, 70, 100)
    plt.plot(x, x, '-r', label='Actual=Prediction', linewidth=1.5)
    plt.axhline(y=10, color='#838B8B', linewidth=1, linestyle='--')
    plt.legend(fontsize=20, loc=2)
    plt.xlabel('Predictions', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.savefig('M:/516/gradientBoosting.png')
    plt.close()


def main():
    decisionTree()
    randomForest()
    featureImp()
    gradientBoosting()


if __name__ == '__main__':
    main()
