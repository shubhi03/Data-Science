#  import the Loading libraries.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#reading and loading datasets in our File Explorer  so we didn't want write full path

data = pd.read_csv('Log-Reg-Case-Study.csv')
#now splliting the dataset.
# in the dependent and independent variable

X = data.ix[:,(1,10,11,13)].values
y = data.ix[:,16]
# info() keyword is used to get the information abt null value and data types
data.info()
# desceribe() keyword is used to get the mathmatical part like mean,std and more of the dataset
# # The head() is used to get the first 5 Rows of the dataset and tail() is to fetch last 5 Rows

data.describe()
data.head()
data.tail()
#importing libraries

from sklearn.linear_model import LogisticRegression

regr = LogisticRegression()
regr.fit(X, y)

pred = regr.predict(X)

#importing libraries forconfusion_matrix to get the Accuracy.

from sklearn.metrics import confusion_matrix

cm_data = pd.DataFrame(confusion_matrix(y, pred).T, index=regr.classes_,columns=regr.classes_)
cm_data.index.name = 'Predicted'
cm_data.columns.name = 'True'
cm_data

model_data = data.sample(frac=0.7)
model_data = data.sample(frac=0.7,random_state=3)
test_data = data.loc[~data.index.isin(model_data.index), :]
data = model_data

#------------------Univariate Analysis--------------------
#----------Capping(Handling the outliers)-----------------


data.Age.describe()
data.Age.quantile(q=0.995)

data = data
min(data.Age)
max(data.Age)
data.loc[data['Age']>75,'Age'] = 75
min(data.Age)
max(data.Age)

data.info()
# *** Missing Vals

data.isnull().values.any()
data.isnull().sum()
data.isnull().sum().sum()



# Housing

pd.crosstab(data.Housing,data.Default_On_Payment)
data.Housing.describe()
data.Housing.unique()
data.Housing.value_counts()

sns.countplot(x='Housing',data=data, palette='hls')
plt.show()
plt.savefig('count_plot')
data.Housing.isnull().sum()
pd.crosstab(data.Housing.isnull(),data.Default_On_Payment)

data['Housing'].mode()
data.Housing[data.Housing=='A152'].count()
data.Housing[data.Housing=='A151'].count()
data.Housing[data.Housing=='A153'].count()
data['Housing'].fillna(data['Housing'].mode()[0],inplace = True)
data.Housing[data.Housing=='A152'].count()
data.Housing.isnull().sum()

'''Bivariate Analysis , Missing Values ,Dummmy Variables'''

pd.crosstab(data.Num_Dependents, data.Default_On_Payment)
data.Num_Dependents.value_counts()
data.Num_Dependents.describe()

data2 = pd.DataFrame()

def get_Percent(col,data):
    grps =data.groupby([col,'Default_On_Payment'])
    print(grps)
    for name, group in grps:
      

        data2.loc[name[0], name[1]] = len(group)

   
    data2['Percentage 0'] = data2[0] * 100 / (data2[0] + data2[1])
    data2['Percentage 1'] = data2[1] * 100 / (data2[0] + data2[1])
    

    (data2.sort_values(by='Percentage 1'))
    # print(df2.sort_values(by='Percentage
    
cols = ['Num_Dependents']
for col in cols:
    get_Percent(col, data)
data.shape
data = data.drop(['Customer_ID','Num_Dependents'],axis=1)
data.shape
      


        '''Job Status'''

data['Job_Status'].unique()
data['Job_Status'].describe()
data['Job_Status'].isnull().sum()
data.Job_Status.value_counts()
sns.countplot(x='Job_Status',data=data, palette='hls')
plt.show()
plt.savefig('count_plot')
data['Job_Status'].mode()
pd.crosstab(data.Job_Status,data.Default_On_Payment)

data['Job_Status'].fillna(data['Job_Status'].mode()[0],inplace = True)
data['Job_Status'].describe()
data['Job_Status'].isnull().sum()


data.info()
data.isnull().sum().sum()
  '''Adv Missing techniques'''



f1 = model_data['Job_Status']=='A171'
f2 = model_data['Job_Status']=='A172'
f3 = model_data['Job_Status']=='A173'
model_data.shape
data['Job_Status'].describe()
(data['Job_Status'].value_counts())
data['Job_Status'].unique()

model_data['Dummy_A171'] = np.where(f1, 1, 0)
model_data['Dummy_A172'] = np.where(f2, 1, 0)
model_data['Dummy_A173'] = np.where(f3, 1, 0)

model_data = model_data.drop(['Job_Status'],axis=1)


cols =['Purpose_Credit_Taken','Status_Checking_Accnt','Credit_History','Job_Status',
       'Years_At_Present_Employment','Marital_Status_Gender',
       'Other_Debtors_Guarantors','Housing','Foreign_Worker']
for col in cols:
    get_Percent(col,data)

(data['Purpose_Credit_Taken'].value_counts())
#get_Percent('Purpose_Credit_Taken',df_base)

data2 = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})

data2['color'] = np.where(data2['Set']=='Z', 'green', 'red')

model_data2 = pd.DataFrame({'Type':list('ABBCDD'), 'Set':list('ZZXYWW')})
(model_data2)


f1 = model_data['Purpose_Credit_Taken']=='P41'
f2 = model_data['Purpose_Credit_Taken']=='P43'
f3 = model_data['Purpose_Credit_Taken']=='P48'

model_data['Dummy_Purpose_Credit_Taken_Low'] = np.where(np.logical_or(f1,np.logical_or(f2,f3)), 1, 0)

data_sbst1 = model_data[['Purpose_Credit_Taken','Dummy_Purpose_Credit_Taken_Low']]
(data_sbst1.head())

f1 = model_data['Purpose_Credit_Taken']=='P49'
f2 = model_data['Purpose_Credit_Taken']=='P40'
f3 = model_data['Purpose_Credit_Taken']=='P45'
f4 = model_data['Purpose_Credit_Taken']=='P50'
f5 = model_data['Purpose_Credit_Taken']=='P46'
#f3 = model_data['Purpose_Credit_Taken']=='P48'
#print(og_data.shape)
model_data['Dummy_Purpose_Credit_Taken_High'] = np.where(np.logical_or(f1,np.logical_or(f2,np.logical_or(f3,np.logical_or(f4,f5)))), 1, 0)
data_sbst1 = model_data[['Purpose_Credit_Taken','Dummy_Purpose_Credit_Taken_Low','Dummy_Purpose_Credit_Taken_High']]


ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
#We'll divide the ages into bins such as 18-25, 26-35,36-60 and 60 and above.
#Understand the output - '(' means the value is included in the bin, '[' means the value is excluded
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats

#Categories (4, object): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]type(cats)
cats[0]
#To include the right bin value, we can do:
pd.cut(ages,bins,right=False)
#Categories (4, object): [[18, 25) < [25, 35) < [35, 60) < [60, 100)]

%matplotlib qt
bins = [0,30,50,100]
ages = model_data.Age
cut =pd.cut(ages,bins,right=True)
pd.crosstab(data.Age,data.Default_On_Payment)
pd.crosstab(data.Age,data.Default_On_Payment).plot(kind='bar')
plt.title(' Frequency of Defaulters')
plt.xlabel('Age')
plt.ylabel('Frequency of Defaults')
#plt.savefig('dflt_fre_job')
plt.show()
#bins = [0,30,50,100]#ages = model_data.Age
#pd.cut(ages,bins,right=True)
#df.loc[df['Age']>75,'Age'] = 75
ages = pd.DataFrame([81, 42, 18, 55, 23, 35], columns=['age'])
bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
ages['agerange'] = pd.cut(ages.age, bins, labels = labels,include_lowest = True)
(ages)

(model_data.shape)
model_data[['Age']].head()
model_data_bkp2 = model_data
bins = [0,30,100]
ages = model_data.Age
lbls = [1,0]
model_data['Dummy_Age_Group'] = pd.cut(ages,labels=lbls,bins=bins)#,right=True)
#pd.cut(ages,[1,0],[0-30,30-100])
(model_data.shape)
model_data[['Age','Dummy_Age_Group']].head()
model_data[['Age','Dummy_Age_Group']].sample(15)
model_data.shape
model_data = model_data.drop(['Age'],axis=1)
model_data.shape



X = model_data.ix[:,(1,2,8,10,13,14,15,16,17)] # ivs for train
X
y = model_data.ix[:,14]
y


#MOdel_Creation

import statsmodels.api as sm

data2 = model_data._get_numeric_data() #drop non-numeric cols
data2.head()
X = data2.loc[:, data2.columns!='Default_On_Payment'].values
#df.drop('b', axis=1)
X
#y = dataset.iloc[:, 4].values
y = data2.iloc[:, 5].values
y

logit = sm.Logit( y, sm.add_constant(X) )
lg = logit.fit()
lg.summary()


def get_significant_vars( lm ):
    var_p_vals_data = pd.DataFrame( lm.pvalues )
    print(var_p_vals_data)
    var_p_vals_data['vars'] = var_p_vals_data.index
    print(var_p_vals_data)
    var_p_vals_data.columns = ['pvals', 'vars']
    print(var_p_vals_data)
    return list( var_p_vals_data[var_p_vals_data.pvals <= 0.05]['vars'] )

significant_vars = get_significant_vars( lg )
significant_vars

X = data2.ix[:,(0,2,4,9)].values
X
logit = sm.Logit( y, sm.add_constant(X) )
lg = logit.fit()
lg.summary()

significant_vars = get_significant_vars( lg )
significant_vars

X = data2.ix[:,(0,1,4)].values
X
logit = sm.Logit( y, sm.add_constant(X) )
lg = logit.fit()
lg.summary()
significant_vars = get_significant_vars( lg )
significant_vars


X = data2.ix[:,(0,2)].values
X
logit = sm.Logit( y, sm.add_constant(X) )
lg = logit.fit()
lg.summary()
significant_vars = get_significant_vars( lg )
significant_vars

X = data2.ix[:,(0,1)].values
X
logit = sm.Logit( y, sm.add_constant(X) )
lg = logit.fit()
lg.summary()
significant_vars = get_significant_vars( lg )
significant_vars

regr = LogisticRegression()
regr.fit(X, y)
print(regr.coef_)
pred = regr.predict(X)
pred

cm_data = pd.DataFrame(confusion_matrix(y, pred).T, index=regr.classes_,columns=regr.classes_)
cm_data.index.name = 'Predicted'
cm_data.columns.name = 'True'
cm_data




          #            THE.........END...........