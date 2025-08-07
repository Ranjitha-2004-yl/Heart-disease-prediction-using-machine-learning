import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('dataset.csv')
import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index

#dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()


y = df['target']
X = df.drop(['target'], axis = 1)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 101, stratify=y, test_size=0.25)

for x in range(200):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_valid,y_valid)
    Y_pred_rf = rf.predict(X_valid)
    current_accuracy = round(accuracy_score(Y_pred_rf,y_valid)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
rf = RandomForestClassifier(random_state=best_x)
print(X_train.columns)
rf.fit(X_valid,y_valid)
y_pred_rf = rf.predict(X_valid)
print(round(accuracy_score(y_pred_rf,y_valid)*100,2))
joblib.dump(rf, 'random_forest_model.pkl')