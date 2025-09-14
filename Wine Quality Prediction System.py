#%% md
# ## A Wine Quality Prediction System
# 
# 
# ### Project Description
# A Wine Quality Prediction System is a machine learning-based system designed to evaluate the quality of wine based on various physicochemical properties. The system uses historical data of wine samples, including features like acidity, alcohol content, pH level, residual sugar, and more, to predict the quality of new wine samples on a predefined scale (usually 1 to 10).
# 
# 
# ### Dataset Description
# Wine Quality Dataset Description
# The Wine Quality Dataset is a well-known dataset available in the UCI Machine Learning Repository. 
# It contains physicochemical attributes of red and white wine samples, 
# along with their quality scores, rated by wine tasters. The dataset is commonly used 
# for regression and classification tasks in machine learning.
# 
# Dataset Overview
# Number of Samples:
# Red wine: 1,599 samples
# Number of Features: 11 (numerical)
# 
# Target Variable: Wine Quality Score (integer value ranging from 0 to 10)
# Type: Multivariate dataset
# 
# Dataset Attributes :-
# Feature	Description	Data Type
# - Fixed Acidity	Concentration of non-volatile acids (e.g., tartaric acid)	Float
# - Volatile Acidity	Concentration of volatile acids (e.g., acetic acid)	Float
# - Citric Acid	Amount of citric acid present (adds freshness)	Float
# - Residual Sugar	Amount of sugar left after fermentation	Float
# - Chlorides	Salt content in the wine	Float
# - Free Sulfur Dioxide	SO₂ available to prevent microbial growth	Float
# - Total Sulfur Dioxide	Total SO₂ (free + bound), used as a preservative	Float
# - Density	Mass per unit volume, affects wine body	Float
# - pH	Measure of acidity or alkalinity	Float
# - Sulphates	Sulfur compounds contributing to antimicrobial properties	Float
# - Alcohol	Alcohol content (% volume)	Float
# - Quality (Target Variable)	Wine quality score (scale 0–10)	Integer
# - Target Variable (Quality Score) Distribution
# 
# The wine quality is rated on a scale of 0 to 10, but most scores range from 3 to 9.
# The majority of wines have quality scores between 5 and 7, 
# making it an imbalanced dataset where extreme quality ratings (0, 1, 9, 10) are rare.
#     
# Dataset Usage
# Regression Task: Predicting the exact quality score as a continuous variable.
# Classification Task: Converting scores into categories (e.g., Low Quality (≤5), Medium (6), High Quality (≥7)).
# Feature Analysis: Understanding how different physicochemical properties affect wine quality
# 
# 
#%%
import pickle

# Step 1 - import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Step 2 - Data Collection
wine=pd.read_csv('winequality-red.csv')
wine.head()
#%%
wine.shape
#%%
# 11 >> features
# 1 >> quality >> target variable
#%%
# Step 3 - Data Cleaning

wine.dtypes
#%%
wine.isnull().sum()
#%%
wine.duplicated().sum()
#%%
wine.drop_duplicates(inplace=True, ignore_index=True)
#%%
wine.shape
#%%
wine['quality'].unique()
#%%
# Step 4 - EDA (ML preprocessing)
# univariate
# bivariate 
# multivariate
#%%
wine.describe()
#%%
wine.groupby('quality').mean()
#%%
wine['quality'].value_counts()
#%%
# dataset is imbalance
#%%
sns.countplot(x=wine['quality'])
plt.show()
#%%
sns.countplot(x=wine['pH'])
plt.show()
#%%
wine['pH'].plot(kind='kde')
#%%
sns.countplot(x=wine['alcohol'])
plt.show()
#%%
sns.kdeplot(wine.query('quality > 2').quality)
#%%
# histogram
wine.hist(figsize=(10,10), bins=50)
plt.show()
#%%
# corr
#%%
plt.figure(figsize=(10,8))
corr=wine.corr()
sns.heatmap(corr, annot=True)
#%%
# ph & fixed_acidity has strong negative correlation
# citric_acid & fixed_acidity has strong positive correlation
# free_sulphur_dioxide & total_sulphur_dioxide has strong positive correlation
#%%
# pair plot
plt.figure(figsize=(11,11))
sns.pairplot(wine)
plt.show()
#%%
# violin plot
sns.violinplot(x='quality', y='alcohol', data=wine)
#%%
# feature selection
# 11 features
#%%
wine['quality'].value_counts()
#%%
wine['goodquality'] = [1 if x >=7 else 0 for x in wine['quality']]
#%%
wine
#%%
X=wine.drop(['quality', 'goodquality'], axis=1)
y=wine['goodquality']
#%%
wine['goodquality'].value_counts()/1359 * 100
#%%
wine.shape
#%%
# feature selection

from sklearn.ensemble import ExtraTreesClassifier
classifier=ExtraTreesClassifier()
#%%
classifier.fit(X, y)
score=classifier.feature_importances_
#%%
s1=pd.Series(score, index=X.columns)
s1.plot(kind='barh')
#%%
# encoding
#%%
# scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_sc=sc.fit_transform(X)
X_sc
#%%
import pickle
with open('wine_sc', 'wb') as f:
    pickle.dump(sc,f)
#%%
# cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_sc, y, test_size=0.2, random_state=7)
#%%
# model training 
#%%
# logistic regression
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1.fit(X_train, y_train)
y_pred=model1.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score', accuracy_score(y_test, y_pred))

confusion_mat=confusion_matrix(y_test, y_pred)
print(confusion_mat)
#%%
# KNN
from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier()
model2.fit(X_train, y_train)
y_pred=model2.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score', accuracy_score(y_test, y_pred))

confusion_mat=confusion_matrix(y_test, y_pred)
print(confusion_mat)
#%%
# Using SVC
from sklearn.svm import SVC
model3=SVC(C=1.0, kernel='rbf')
model3.fit(X_train, y_train)
y_pred=model3.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score', accuracy_score(y_test, y_pred))

confusion_mat=confusion_matrix(y_test, y_pred)
print(confusion_mat)
#%%
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4=DecisionTreeClassifier(criterion='entropy', random_state=7)
model4.fit(X_train, y_train)
y_pred=model4.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score', accuracy_score(y_test, y_pred))

confusion_mat=confusion_matrix(y_test, y_pred)
print(confusion_mat)
#%%
# Random Forest
from sklearn.ensemble import RandomForestClassifier
model5=RandomForestClassifier(random_state=1, n_estimators=50)
model5.fit(X_train, y_train)
y_pred=model5.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score', accuracy_score(y_test, y_pred))

confusion_mat=confusion_matrix(y_test, y_pred)
print(confusion_mat)
#%%
# Xgboost
import xgboost as xgb
model6=xgb.XGBClassifier(random_state=11)
model6.fit(X_train, y_train)
y_pred=model6.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score', accuracy_score(y_test, y_pred))

confusion_mat=confusion_matrix(y_test, y_pred)
print(confusion_mat)
#%%
# Navie Bays
from sklearn.naive_bayes import GaussianNB
model7=GaussianNB()
model7.fit(X_train, y_train)
y_pred=model7.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score', accuracy_score(y_test, y_pred))

confusion_mat=confusion_matrix(y_test, y_pred)
print(confusion_mat)
#%%
results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [0.90,0.88,0.89,0.83,0.86,0.88,0.88]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
#%%
# Hence we will use Logistic Regression for training model
# model1
#%%
with open('wine_model','wb') as f:
    pickle.dump('model1', f)
#%%
# prediction
a=[7.4,0.700,0.00,1.9,0.076,11.0,34.0,0.99780,3.51,0.56,9.4]
a_sc=sc.transform([a])
#%%
model1.predict(a_sc)