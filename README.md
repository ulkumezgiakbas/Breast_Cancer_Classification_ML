# Breast Cancer Classification with SVM

This project applies machine learning techniques to classify breast cancer cases using the **Breast Cancer Wisconsin Dataset**. The classification is done using **Support Vector Machines (SVM)**, and hyperparameter tuning is performed with **GridSearchCV**.

## 📌 Dataset Information
- Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- The dataset is loaded from `sklearn.datasets.load_breast_cancer()`.
- It contains **30 features** and **1 target variable** (malignant or benign).
- Shape of the dataset: `(569, 30)`

## 🚀 Steps in the Project
1. **Load & Explore Data** 📊
2. **Convert Data to Pandas DataFrame** 📝
3. **Visualizations** 📈
4. **Train-Test Split** ✂️
5. **Train an SVM Model** 🤖
6. **Optimize Model with GridSearchCV** 🔍
7. **Evaluate & Interpret Results** 🏆

## 🔬 Implementation

### 1️⃣ Load Data
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

cancer = load_breast_cancer()
df_cancer = pd.DataFrame(
    np.c_[cancer['data'], cancer['target']],
    columns=np.append(cancer['feature_names'], 'target')
)
```

### 2️⃣ Exploratory Data Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df_cancer, hue='target', vars=['mean radius', 'mean texture', 'mean area'])
sns.heatmap(df_cancer.corr(), annot=True, cmap='coolwarm')
plt.show()
```

### 3️⃣ Train-Test Split
```python
from sklearn.model_selection import train_test_split
x = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
```

### 4️⃣ Train SVM Model
```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(x_train, y_train)
y_pred = svc_model.predict(x_test)
print(classification_report(y_test, y_pred))
```

### 5️⃣ Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(x_train, y_train)
print(grid.best_params_)
```

### 6️⃣ Final Predictions & Heatmap
```python
import seaborn as sns
cm = confusion_matrix(y_test, grid.predict(x_test))
sns.heatmap(cm, annot=True, fmt='d')
```

## 🎯 Results
- The best hyperparameters were found using `GridSearchCV`.
- The final model achieves **high accuracy in predicting breast cancer cases**.
- The visualization provides insights into feature correlations.

## 📌 Next Steps
✅ Try other ML models like **Random Forest, Logistic Regression**.  
✅ Improve feature selection & engineering.  
✅ Deploy the model using Flask or FastAPI.

![myplot](https://github.com/user-attachments/assets/4d9cb3f5-33a8-41d6-bc47-bccc3c7e5b05)
![myplot2](https://github.com/user-attachments/assets/639cb304-08f8-4328-9767-024a4ef23158)


---

🔗 LinkedIn: https://www.linkedin.com/in/uezgiakbas/



