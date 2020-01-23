#Loading important libraries.

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import os

#reading and loading datasets in our File Explorer  so we didn't want write full path

data=pd.read_csv("MFGEmployees4.csv")

# The head() is used to get the first 5 Rows of the dataset
data.head()
# The shape is used to get the dimensions of the dataset
data.shape
# desceribe() keyword is used to get the mathmatical part like mean,std and more of the dataset
data.describe()
# info() keyword is used to get the information abt null value and data types
data.info()

#check the types of datasets.
data.dtypes

# tail()is opposite of head and fetch last 5 rows.
data.tail()

# Summarise the non-numerical data
data.describe(include=['O'])

#Data pre-processing.

# Look at full list of surname and frequency
data.Surname.value_counts()

# Look at full list of givenNmae and frequency
data.GivenName.value_counts()

# Look at full list of Gender and frequency
data.Gender.value_counts()

# Look at full list of CIty and frequency
data.City.value_counts()

# Look at full list of JobTitle and frequency
data.JobTitle.value_counts()

# Look at full list of DepartmentName and frequency
data.DepartmentName.value_counts()

# Look at full list of StoreLoacation and frequency
data.StoreLocation.value_counts()

# Look at full list of Division and frequency
data.Division.value_counts()

# Look at full list of Age and frequency
data.Age.value_counts()

# Look at full list of LengthService and frequency
data.LengthService.value_counts()

# Look at full list of AbsentHours and frequency
data.AbsentHours.value_counts()

# Look at full list of BusinessUnit and frequency
data.BusinessUnit.value_counts()

#Counting missing values for different columns.
data.isnull().sum()

data.columns

#information about original Attrition datasets.
data.info()

data.isna().sum()

#Histogram for numeric data for Absenteeism Datasets.

plt.figure(figsize = (9, 5)) 
data['EmployeeNumber'].plot(kind ="hist")

#Now create correlation between numeric columns of absenteeism datasets.

#Checking for correlation among all the x(inputs)

#kendall--> is a statistic used to measure the ordinal association between two measured quantities.
corr = data.corr(method='kendall')

#heatmap gives co-relation between two numeric  variables.

plt.figure(figsize=(8,4))
sns.heatmap(corr, annot=True)

print(corr)

#Grid Correlation matrix.

corrmat = data.corr() 
cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) #change the colour of ytick.
cg

#now correlation between textual to textual columns of absenteeism datasets.

data['Surname']=data['Surname'].astype('category').cat.codes
data['GivenName']=data['GivenName'].astype('category').cat.codes
data['Gender']=data['Gender'].astype('category').cat.codes
data['City']=data['City'].astype('category').cat.codes
data['JobTitle']=data['JobTitle'].astype('category').cat.codes
data['DepartmentName']=data['DepartmentName'].astype('category').cat.codes
data['StoreLocation']=data['StoreLocation'].astype('category').cat.codes
data['Division']=data['Division'].astype('category').cat.codes
data['BusinessUnit']=data['BusinessUnit'].astype('category').cat.codes
data.corr()
data.info()

#now some visualiation parts begins.

np.random.seed(1234)
df = pd.DataFrame(np.random.randn(10,4),columns=['EmployeeNumber', 'Surname', 'GivenName', 'Gender'])
boxplot = df.boxplot(column=['EmployeeNumber', 'Surname', 'GivenName', 'Gender'])

np.random.seed(1234)
df = pd.DataFrame(np.random.randn(10,9),columns=['City', 'JobTitle', 'DepartmentName', 'StoreLocation','Division','Age','LengthService','AbsentHours','BusinessUnit'])
boxplot = df.boxplot(column=['City', 'JobTitle', 'DepartmentName', 'StoreLocation','Division','Age','LengthService','AbsentHours','BusinessUnit'])

#now data preparing part begins.

#sum of all the values of AbsentHours Columns.
data.at['Total','AbsentHours']=data['AbsentHours'].sum()
data.tail()

data.columns

#now i filter the age columns.

data = data.loc[data["Age"]>=18]
data = data.loc[data["Age"]<=65]
data.head()
data.tail()
data.shape

#now create new columns name= AbsentRate.
data['AbsentRate'] = data.AbsentHours/2080*100
data.head()

#now to remove some unwanted columns which doesnt have use in model frm dataset
data = data.drop(columns = ['EmployeeNumber', 'Surname', 'GivenName','StoreLocation','City','JobTitle','DepartmentName','AbsentHours'])
data.head()

data1 = data.drop(columns = ['Division'])
data1.head()
data1.shape

#now i'm splliting the apsenteeism dataset.
# in the dependent and independent variable

X=data1.iloc[:,[0,1,2,3]]
y=data1['AbsentRate']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=353)

#now encoder part start.

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from numpy import array

#labelencoder=LabelEncoder()
#X.iloc[:,0]=labelencoder.fit_transform(X[:,0])
#X

#labelencoder=LabelEncoder()
#X.iloc[:,3]=labelencoder.fit_transform(X[:,3])
#X

# no we apply some algorithms.

# 1. Linear Regression Algorithms.

regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
r2_score(y_test,y_pred)

#2. Decesion TRee.

from sklearn import tree
model1=tree.DecisionTreeRegressor()
model1.fit(X_train,y_train)
model1.score(X_train,y_train)

#3. Random Forest Algorithm.

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 1)
rf.fit(X_train, y_train)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(X_test,y_test)*100))

# 4. KNN.

from sklearn.neighbors import KNeighborsClassifier
model2= KNeighborsClassifier(n_neighbors=5)
model1.fit(X_train,y_train)
model1.score(X_train,y_train)


