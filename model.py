# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:55:05 2020

@author: admin
"""

"importing files for the model"
import pickle
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
'exec(%matplotlib inline)'

"Dataset downloading"
dataset=pd.read_csv('C:/Users/admin/Desktop/deployment/Diabties.csv')
dataset.isnull().sum()

"Checking for imbalance dataset"
diabetes_positive=len(dataset[dataset['Outcome']==1])
diabetes_negative=len(dataset[dataset['Outcome']==0])
print(("Positive Percentage:{0}".format(int((diabetes_positive/dataset['Outcome'].count())*100))))
print(("Negative Percentage:{0}".format(int((diabetes_negative/dataset['Outcome'].count())*100))))

"Heatmap to calculate the correlation(figure_1)"
corr = dataset.corr()
print(corr)
sns.heatmap(corr, 
         xticklabels=corr.columns, 
         yticklabels=corr.columns)

"Missing values in the dataset"
print('total number of rows:{0}'.format(len(dataset)))
print("number of rows missing glucose:{0}".format(len(dataset.loc[dataset['Glucose']==0])))
print("number of rows missing BloodPressure:{0}".format(len(dataset.loc[dataset['BloodPressure']==0])))
print("number of rows missing SkinThickness:{0}".format(len(dataset.loc[dataset['SkinThickness']==0])))
print("number of rows missing Insulin:{0}".format(len(dataset.loc[dataset['Insulin']==0])))
print("number of rows missing Age:{0}".format(len(dataset.loc[dataset['Age']==0])))
print("number of rows missing Pregnancies:{0}".format(len(dataset.loc[dataset['Pregnancies']==0])))

"Checking for Outliers using boxblot(figure_2)"
feature_columns=[dataset['BloodPressure'],dataset['SkinThickness'],dataset['Age'],dataset['Glucose']]
fig = plt.figure(1, figsize=(10,8))
ax = fig.add_subplot(111)
Ps = ax.boxplot(feature_columns)

"Age Distribution barchart(figure_3)"
bins = [20,25,30,35,40,45,50,55,60,65,70,75,80]
plt.hist(dataset['Age'], bins, histtype='bar', rwidth=0.5)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('No of Persons')
plt.show()

"Filling the missing values with mean values"
dataset['Insulin']=dataset['Insulin'].replace(0,int(dataset['Insulin'].mean()))
dataset['SkinThickness']=dataset['SkinThickness'].replace(0,int(dataset['SkinThickness'].mean()))

"Spliting data into train and test split"
features_columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
predicted_class=['Outcome']
x=dataset[features_columns].values
y=dataset[predicted_class].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

"LogisticRegression model for prediction"
model=LogisticRegression(max_iter=300)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

"Calculating accuracy score and confusion matrix"
print(accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

"Roc and Auc curve for finding Threshold value(figure_4)"
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic')
#pyplot.scatter(0.5,0.78, marker='o', color='black', label='Best')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.show()

"Calculating precision and recall(figure_5)"
# calculate pr-curve
precision, recall, thresholds = precision_recall_curve(y_test,y_pred)
# plot the roc curve for the model
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
plt.show()

"Hyperparameter turning for logistic regression"
max_iter=[100,200,300,400,500,600]
solver=['newton-cg','lbfgs','liblinear','sag','saga']
penalty=['l1','l2','elasticnet','none']
fit_intercept=['True','False']
C=[x for x in np.linspace(0.5,4.0,8)]
l1_ratio=[float(x)for x in np.linspace(start=0,stop=1,num=10)]

"RandomizedsearchCV"
random_grid={'penalty':penalty,
             'C':C,
             'max_iter':max_iter,
             'solver':solver,
             'fit_intercept':fit_intercept,
              'l1_ratio':l1_ratio}
print(random_grid)

"fitting the randomizedsearchcv"
lr=LogisticRegression()
lr_random=RandomizedSearchCV(estimator=lr,param_distributions=random_grid,n_iter=10,cv=5,verbose=2,random_state=100,n_jobs=1)
lr_random.fit(x_train,y_train.ravel())

"best parameter for the model"
lr_random.best_params_
best_random_grid=lr_random.best_estimator_

"Report to Randomizedsearchcv"
y_pred=best_random_grid.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score{}".format((accuracy_score(y_test,y_pred))))
print("Classification report:{}".format(classification_report(y_test,y_pred)))

"saving model as a disk"
pickle.dump(model,open('prediction.pkl','wb'))

"loding model to compare the results"
prediction =pickle.load(open('prediction.pkl','rb'))
print(model.predict([[7, 150, 78, 29, 126, 35.2, 0.692, 54]]))






