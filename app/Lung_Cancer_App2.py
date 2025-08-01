#### 1. Importing Libraries ####

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
plt.style.use('fivethirtyeight')
colors = ['#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0']
sns.set_palette(sns.color_palette(colors))

df = pd.read_csv("survey lung cancer.csv")
df.head()

#### 2. Understanding Our Data ####

# What is the shape of the dataset?
df.shape

# Some info about our attributes and its datatype
df.info()

# Some analysis on the numerical columns
df.describe()

# Check for null values
df.isnull().sum()

# Check for duplicates in the dataset
df.duplicated().sum()

# Dropping duplicate rows from the dataset
df.drop_duplicates(inplace=True)

# Shape of dataset after duplicate rows are dropped
df.shape


# A short preprocessing step before we move on to Exploratory Data Analysis for easy implementation of graphs. Here we are encoding LUNG_CANCER and GENDER column.
# Basically replacing the values in the columns to 0, 1

encoder = LabelEncoder()
df['LUNG_CANCER'] = encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = encoder.fit_transform(df['GENDER'])
df.head()

# separating continuous and categorical columns
con_col = ['AGE']
cat_col = []
for i in df.columns:
    if i != 'AGE':
        cat_col.append(i)


X = df.drop(['LUNG_CANCER'], axis=1)
y = df['LUNG_CANCER']

for i in X.columns[2:]:
    temp = []
    for j in X[i]:
        temp.append(j-1)
    X[i] = temp
X.head()

X_over, y_over = RandomOverSampler().fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_over, y_over, random_state=42, stratify=y_over)
print(f'Train shape : {X_train.shape}\nTest shape: {X_test.shape}')

###################
age_mean = X_train["AGE"].mean()
print(age_mean)

age_sd = X_train["AGE"].std()
print(age_sd)

###################

scaler = StandardScaler()
X_train['AGE'] = scaler.fit_transform(X_train[['AGE']])
X_test['AGE'] = scaler.transform(X_test[['AGE']])
X_train.head()


# SVM Tutorial
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC() # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# giving inputs to the machine learning model
# features = [[sepal_length, sepal_width, petal_length, petal_width]]
age=int(input("enter age"))
g=(age-age_mean)/(age_sd)
print(g)
a=15
features = np.empty(a)
print("Enter in order: gender, age, smoking, yellow fingers, anxiety, peer pressure, chronic disease, fatigue, allergy, wheezing, alcohol, coughing, shortness breath, swallowing, chest pain")
for i in range (len(features)):
    x=float(input('Enter Elements'))
    features[i]=x
print(np.floor(features))
features1=[features]
# using inputs to predict the output
prediction = clf.predict(features1)
print("Prediction: {}".format(prediction))
