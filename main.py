import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.keys())
print(cancer['DESCR'])
print(cancer['target_names'])
print(cancer['target'])
print(cancer['feature_names'])
print(cancer['data'])

data_shape = cancer['data'].shape
print(data_shape)

df_cancer = pd.DataFrame(
    np.c_[cancer['data'], cancer['target']],
    columns=np.append(cancer['feature_names'], 'target')
)

print(df_cancer.head())
print(df_cancer.tail())

sns.pairplot(df_cancer, hue='target',
             vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(x='target', data=df_cancer, label="Count")

sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df_cancer)

plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot=True)

plt.show()

x = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

min_train = x_train.min()
range_train = (x_train - min_train).max()
x_train_scaled = (x_train - min_train) / range_train

min_test = x_test.min()
range_test = (x_test - min_test).max()
x_test_scaled = (x_test - min_test) / range_test

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(x_train_scaled, y_train)

y_predict = svc_model.predict(x_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True, fmt="d")
print(classification_report(y_test, y_predict))

sns.scatterplot(x=x_train['mean area'], y=x_train['mean smoothness'], hue=y_train)
sns.scatterplot(x=x_train_scaled['mean area'], y=x_train_scaled['mean smoothness'], hue=y_train)

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)
grid.fit(x_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(x_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, grid_predictions))
