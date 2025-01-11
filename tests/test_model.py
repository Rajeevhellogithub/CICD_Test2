import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import itertools

sns.set(style='white')

# Test 1: Data Loading and Preprocessing
def test_data_loading():
    dataset = pd.read_csv('tests/iris.csv')
    dataset.columns = [colname.strip(' (cm)').replace(" ", "_") for colname in dataset.columns.tolist()]
    assert not dataset.empty, "Dataset is empty"
    return dataset

# Test 2: Feature Engineering
def test_feature_engineering():
    dataset = test_data_loading()
    dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']
    assert 'sepal_length_width_ratio' in dataset.columns
    assert 'petal_length_width_ratio' in dataset.columns
    return dataset

# Test 3: Train-Test Split
def test_train_test_split():
    dataset = test_feature_engineering()
    dataset = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 
                       'sepal_length_width_ratio', 'petal_length_width_ratio', 'target']]
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=44)
    assert not train_data.empty, "Train data is empty"
    assert not test_data.empty, "Test data is empty"
    return train_data, test_data

# Test 4: Model Training - Logistic Regression
def test_logistic_regression():
    train_data, test_data = test_train_test_split()
    X_train = train_data.drop('target', axis=1).values.astype('float32')
    y_train = train_data['target'].values.astype('int32')
    X_test = test_data.drop('target', axis=1).values.astype('float32')
    y_test = test_data['target'].values.astype('int32')
    
    logreg = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=100, multi_class='multinomial')
    logreg.fit(X_train, y_train)
    predictions = logreg.predict(X_test)
    
    cm = confusion_matrix(y_test, predictions)
    assert cm.shape[0] == 3, "Confusion matrix size mismatch"
    return logreg, X_train, y_train, X_test, y_test, cm

# Test 5: Model Training - Random Forest
def test_random_forest():
    train_data, test_data = test_train_test_split()
    X_train = train_data.drop('target', axis=1).values.astype('float32')
    y_train = train_data['target'].values.astype('int32')
    X_test = test_data.drop('target', axis=1).values.astype('float32')
    y_test = test_data['target'].values.astype('int32')
    
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_train)
    predictions = rf_reg.predict(X_test)
    predictions_class = np.round(predictions).astype(int)
    
    assert len(predictions_class) == len(y_test), "Prediction length mismatch"
    return rf_reg, predictions_class, y_test

# Test 6: Confusion Matrix Plot
def test_plot_confusion_matrix():
    _, _, _, _, y_test, cm = test_logistic_regression()
    target_name = np.array(['setosa', 'versicolor', 'virginica'])
    assert cm.sum() == len(y_test), "Mismatch in confusion matrix total"
    return cm

# Test 7: Feature Importance Plot
def test_feature_importance():
    rf_reg, _, _ = test_random_forest()
    importances = rf_reg.feature_importances_
    assert len(importances) > 0, "Feature importances not computed"
    return importances

# Test 8: Save Scores
def test_save_scores():
    logreg, X_train, y_train, X_test, y_test, _ = test_logistic_regression()
    rf_reg, _, _ = test_random_forest()
    
    scores = {
        "Random Forest Train Acc": rf_reg.score(X_train, y_train) * 100,
        "Logistic Regression Train Acc": logreg.score(X_train, y_train) * 100
    }
    assert scores["Random Forest Train Acc"] > 0, "Random Forest accuracy is invalid"
    assert scores["Logistic Regression Train Acc"] > 0, "Logistic Regression accuracy is invalid"
    return scores
