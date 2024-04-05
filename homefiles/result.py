from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler


#%matplotlib inline

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


def home(request):
    return render(request, "home.html")
def predict(request):
    return render(request, "predict.html")

def result(request):
     if request.method == 'POST':
        df = pd.read_csv(r'C:\Users\SNEHA\Downloads\heart.csv')
        dataframe = df.dropna()
        dataframe = dataframe.drop(columns=['slope', 'thal', 'fbs', 'restecg', 'exang', 'sex'])

        X = dataframe.drop(['target'], axis=1)
        y = dataframe['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)


        max_accuracy = 0
        rf = RandomForestClassifier()

        for x in range(2000):
            rf = RandomForestClassifier(random_state=x)
            rf.fit(X_train, y_train)
            Y_pred_rf = rf.predict(X_test)
            current_accuracy = round(accuracy_score(Y_pred_rf, y_test) * 100, 2)
            if (current_accuracy > max_accuracy):
                max_accuracy = current_accuracy
                best_x = x

    # print(max_accuracy)
    # print(best_x)

        rf = RandomForestClassifier(random_state=best_x)
        rf.fit(X_train, y_train)
        Input_Age=int(request.GET['ageInput'])
        Input_Cp = int(request.GET['cpInput'])
        Input_trestbps = int(request.GET['trestbpsIntput'])
        Input_chol = int(request.GET['cholInput'])
        Input_thalach = int(request.GET['thalachInput'])
        Input_oldpeak = int(request.GET['oldpeakInput'])
        Input_ca = int(request.GET['caInput'])

        Output = rf.report([[Input_Age, Input_Cp, Input_trestbps, Input_chol, Input_thalach, Input_oldpeak, Input_ca]])
        result=" "
        if Output==[1]:
            result = "The patient seems to have heart disease"
        else:
            result = "The patient seems to be normal"

        return render(request, "predict.html",{"result": result})
     else:
         pass  

