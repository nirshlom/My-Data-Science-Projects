# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:14:29 2018

@author: nirsh
"""
# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn import preprocessing
from tableone import TableOne
import os

# get the data: CHF 30 selected variables N = 7169
path_wd = r'C:\Users\nirsh\Dropbox\Python\CHF_PROJECT\DECISION_TREE\progs'
os.chdir(path_wd)
path_file = r'C:\Users\nirsh\Dropbox\Python\CHF_PROJECT\DECISION_TREE\files\CHF_30_vars.csv'
dat_main = pd.read_csv(path_file, index_col = 'Patient')

#exploring the data
dat_main.info() #like str function in r

#data preperation
#dat_chf['gender'] = np.where(dat_chf['gender'] == 1, 'male', 'female')
list_data_features = list(dat_main)
list_my_features = ['CHF60', 'age', 'gender', 'ECHO1_ef', 'pre_PCI',
                    'pre_MI', 'pre_CHF', 'pre_CVA', 'pre_PVD', 'pre_DYSLIP',
                    'pre_HTN', 'pre_DM', 'pre_COPD', 'FIRST_HGB', 'LV_FUNCTION',
                    'SPAP', 'GFR_MDRD', 'SODIUM', 'HCT', 'ACE_ARB', 'ANY_ALDACTONE',
                    'HOS_DURATION', 'PRE_AFIB', 'SBP', 'DBP', 'GLUCOSE', 'SGOT', 'FIRST_INR',
                    'BMI', 'WEIGHT', 'WEIGHT', 'LV_MASS_INDEX']
#check for missing or NaN in the dataset:
pd.isnull(dat_main).sum() > 0

#dataset for analysis outcome = chf 60
dat_chf = dat_main[list_my_features].copy() #the copy() is important to create new dataframe
dat_chf.head()
describe = dat_chf.describe() #function for easy descriptive statistics

# A simple way to fill na (one by one)
median_glu = dat_chf['GLUCOSE'].median()
dat_chf['GLUCOSE'] = dat_chf['GLUCOSE'].fillna(median_glu) 

#imputer sklearn (better option to fill missing values) for all df
imputer = Imputer(strategy = 'median', axis = 1)
dat_chf = pd.DataFrame(imputer.fit_transform(dat_chf), columns = dat_chf.columns) # the imputation

pd.isnull(dat_chf).sum() > 0

#descriptive statistics (table 1)
columns = ["age", "gender", "ECHO1_ef", "pre_MI","pre_DM"]
categorical = ['gender', 'pre_MI', 'pre_DM']
groupby = ["CHF60"]
labels={'ECHO1_ef': 'Ejection fraction',
        'pre_MI': 'Previous MI',
        'CHF60' : 'CHF 60 days'}
mytable = TableOne(dat_chf, columns = columns,
                   categorical = categorical,
                   groupby = groupby,
                   labels = labels,
                   isnull = True, remarks = False, pval = True)

print(mytable)

#data visualization:




#Preprocessing:
#scale the data
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(dat_chf.drop('CHF60', axis = 1))
dat_chf_scaled = pd.DataFrame(scaled_df, columns = dat_chf.columns[1:])
dat_chf_scaled_final = pd.concat((dat_chf_scaled, dat_chf['CHF60']), axis =1)

#split train and test
chf_train, chf_test = train_test_split(dat_chf, test_size=0.3, random_state=12345)
chf_train.head()

#Model 1: decision tree

#step 2: function to bulild and evaluate the model
def evaluate_model(**kwargs):

    # Fit the model
    X = chf_train.drop('CHF60', axis=1)
    y = chf_train.CHF60 
    chf_dt = DecisionTreeClassifier(**kwargs).fit(X, y)
    
    #export to graphviz
    export_graphviz(decision_tree= chf_dt,
                    out_file='chf_dt.dot',
                    feature_names=X.columns,
                    class_names=['0', '1'],
                    leaves_parallel=True,
                    filled=True,
                    rotate=False,
                    rounded=True)
    
    # Assess the model
    train_acc = accuracy_score(
        y_true=chf_train.CHF60,
        y_pred=chf_dt.predict(X))
    test_acc = accuracy_score(
        y_true=chf_test.CHF60,
        y_pred=chf_dt.predict(chf_test.drop('CHF60', axis=1)))
    
    print("Train accuracy: {:.2f}".format(train_acc))
    print("Test accuracy : {:.2f}".format(test_acc))

#step 3: evaluate the model
evaluate_model(max_depth=1, min_samples_split=10, min_samples_leaf=10)
#evaluate_model(max_depth=4, min_samples_split=10, min_samples_leaf=10)
#evaluate_model(max_depth=4, min_samples_split=50, min_samples_leaf=10)
#evaluate_model(max_depth=50, min_samples_split=2, min_samples_leaf=1)

#export predicted probability:
pred_probas = pd.DataFrame(chf_dt.predict_proba(X),
                           columns=chf_dt.classes_)
dat_chf_pred = pd.concat((dat_chf, pred_probas[1]), axis =1)
     
dat_chf_pred.rename(columns = {1.0 : 'predicted_dt'}, inplace = True)
list(dat_chf_pred)

#############################################################################

#Model 2: Logistic regression:
X = chf_train.drop('CHF60', axis=1)
y = chf_train.CHF60 
pd.isnull(chf_train).sum() > 0
#build the model
chf_lr = LogisticRegression().fit(X, y)
#train and predict
chf_train['predict_lr'] = pd.Series(chf_lr.predict(X))

#Cofusion matrix
conf_matrix_lr = confusion_matrix(y_true = chf_train['CHF60'],
                                  y_pred = chf_train['predict_lr'])
pd.DataFrame(conf_matrix_lr,
             index = chf_lr.classes_,
             columns = chf_lr.classes_)

#Classification report
print(classification_report(y_true = chf_train['CHF60'],
                                  y_pred = chf_train['predict_lr']))

#validation

#############################################################################
#Model 3: KNN
#build the model
chf_knn = KNeighborsClassifier(n_neighbors=5, metric='cosine',
                                algorithm='brute')
chf_knn.fit(X, y)

#Asses the model
train_pred_knn = chf_knn.predict(X)
conf_matric_knn_train = confusion_matrix(y_true = y,
                                   y_pred = train_pred_knn)
pd.DataFrame(conf_matric_knn,
             index = chf_knn.classes_,
             columns = chf_knn.classes_)

#calssification report:
print(classification_report(y_true = y,
                            y_pred = train_pred_knn))

#Change hyperparameters
accuracies = []

for k in range(1, 10):
    chf_knn = KNeighborsClassifier(n_neighbors = k,
                                   metric = 'euclidean',
                                   algorithm = 'brute').fit(X, y)
    accuracies.append(accuracy_score(y_true = y,
                                     y_pred = chf_knn.predict(X)))
plt.plot(accuracies)    






##############################################################################
#############################TEST THE 3 Models###############################

#Decision tree

#logistic regression




#KNN
#validation (on test data)
    X = chf_test.drop('CHF60', axis=1)
    y = chf_test.CHF60 
test_pred_knn = chf_knn.predict(X)

#assess the test data
conf_mat_knn_test = confusion_matrix(y_true = y,
                                     y_pred = test_pred_knn)
pd.DataFrame(conf_mat_knn_test,
             index = chf_knn.classes_,
             columns = chf_knn.classes_)
#classifiaction reort on test
print(classification_report(y_true = y,
                            y_pred = test_pred_knn))   