import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mode_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matric, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

if not os.path.exists('sklearn_figures'):
    os.makedirs('sklearn_figures')

df = pd.read_cvs("iris.csv")
X = df.drop('class', axis = 1)
Y = df['class']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_cm = confusion_matric(y_test, lr.predict(X_test))
ConfusionMatrixDisplay(lr_cm, display_labels=label_encoder.classes_).plot()
plt.title('Confusion-Matrix-for-Logisitic-Regression')
plt.savefig('sklearn_figures/logistic_regression_cm.png')
plt.close()

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train[['sepal_length', 'sepal_width']], y_train)
xx, yy = np.meshgrid(
    np.linspace(X['sepal_length'].min(),
        X['sepal_length'].max(), 100),
    np.linspace(X['sepal_width'].min(),
        X['sepal_width'].max(), 100))
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
for label in np.unique(y_encoded):
    plt.scatter(X_train['sepal_length'][y_train == label])
plt.title('Decision-Boundary-for-Decision-Tree')
plt.xlabel('Sepal-Length')
plt.ylabel('Sepal-Width')
plt.legend()
plt.savefig('sklearn_figures/decision_tree_boundary.png')
plt.close()

# Random Forest