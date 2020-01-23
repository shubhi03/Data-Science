'''This notebook will go through making 3 machine learning models to predict 
employment termination based on knowledge of an employees work in the company.'''
# =============================================================================
#importing libraries
# =============================================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
os.getcwd()
os.chdir('C:\\Users\\HP\\Desktop\\employee-attrition-data')
# =============================================================================
#importing dataset
# =============================================================================
dataset=pd.read_csv('MFG10YearTerminationData.csv')

# =============================================================================
#quick summary of the data
# =============================================================================
dataset.info() #show total nos. of column/rows,datatype
# 18 columns 49653 rows ,no null value
dataset.describe() # describe numerical columns indataset
# 5 numeric columns
dataset.describe(include=['O'])# describe non numerical columns in dataset
# 13 object data type 
dataset.count() # gives total count/entry in column 
# all columns having 49653 values
dataset.describe(include='all') # describe all numeric & text column details
# =============================================================================
#More Detailed 
# =============================================================================
#Exploratory

# age barplot
dataset['age'].value_counts().sort_index().plot(kind='bar')
dataset.age.value_counts().sort_index()
# minimum age 19, maxm age 65
#as we have have age column we may remove birthdate_key column 

# length_of_service barchart
dataset['length_of_service'].value_counts().plot(kind='bar',figsize=(10, 8))
dataset.length_of_service.value_counts().sort_index()
#minimum service is 0, maximum service is 26 years
#as we have length_of_service column we may remove orghiredata_key

# city_name barchart
dataset['city_name'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.city_name.value_counts()
dataset.city_name.nunique() 
# total 40 unique city 

#  department_name barplot
dataset['department_name'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.department_name.value_counts()
dataset.department_name.nunique()
#total 21 department names

#  department_name barplot
dataset['job_title'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.job_title.value_counts()
dataset.job_title.nunique()
#total 47 job titles

# store_name bar chart
dataset['store_name'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.store_name.value_counts().sort_index()
dataset.store_name.nunique()
#total 46 store names

# gender_short/gender_full bar chart
dataset['gender_short'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.gender_short.value_counts()
dataset.gender_short.nunique()
dataset['gender_full'].value_counts().plot(kind='bar',figsize=(8,6))
dataset.gender_full.value_counts()
dataset.gender_full.nunique()
#2 genders male,female
#gender_short & gender_full having same data so we may remove one of them

#termreason_desc bar chart
dataset['termreason_desc'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.termreason_desc.value_counts()
dataset.termreason_desc.nunique()
#4 reasons NA,retirement,resignation,layoff

#termtype_desc bar chart
dataset['termtype_desc'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.termtype_desc.value_counts()
dataset.termtype_desc.nunique()
# 3 types NA,voluntary,involuntary

# STATUS_YEAR bar chart
dataset['STATUS_YEAR'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.STATUS_YEAR.value_counts().sort_index()
dataset.STATUS_YEAR.nunique()
#10 years #2006 to 2015

# STATUS bar chart
dataset['STATUS'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.STATUS.value_counts()
dataset.STATUS.nunique()
#2 STATUS, ACTIVE & TERMINATED

# BUSINESS_UNIT bar chart
dataset['BUSINESS_UNIT'].value_counts().plot(kind='bar',figsize=(8, 6))
dataset.BUSINESS_UNIT.value_counts()
dataset.BUSINESS_UNIT.nunique()
#2 BUSINESS UNIT, STORE & HEADOFFICE


#Checking for correlation among all numeric columns
#kendall--> is a statistic used to measure the ordinal association between two measured quantities.
corr = dataset.corr(method='kendall')
corr
#heatmap gives co-relation between two numeric  variables.
plt.figure(figsize=(8,4))
sns.heatmap(corr, annot=True)
print(corr)

# =============================================================================
#SOME OBSERVATIONS FROM ABOVE ANALYSIS
#1.  Data is a mix of string and integer values.
#2.  Some features with strings should be dates. These should be converted to date and time if used. The features are recorddate_key, birthdate_key, orighiredate_key, and terminationdate_key.
#3.  EmployeeID is for identification. It shouldn't be used for training the machine learning model, but can be useful for filtering rows.
#4.  The age can be found by usng the record date and the birth date. So one set may be dropped.
#5.  The length of service can be found using the record date and the original hire date. So one set may be dropped.
#6.  The termination date uses 1/1/1900 if the employee is still active.
#7.  The store_name is given as a number, even though it is a nominal categorical feature. The store name itself is unlikely to be cause of employment termination, but particular feature values may be associated with particular stores. It could be an interesting separate investigation.
#8.  Gender is given in short and full. Only one of them is necessary so one will be dropped.
#9.  An employee whose employment is terminated has valid entries for termination date, termination reason and termination type. These 3 features should not be used for training the machine learning model because the features are results, not predictors, of employment termination. They may be interesting as labels, however, if the prediction goal changes.
#10. The status_year column repeats information in the record date.
#11. The status column is the label to predict. It should be converted from string to numerical.
# =============================================================================
# Create new categories for job titles
# Look at full list of job titles and frequency
dataset.job_title.value_counts()
dataset.job_title.nunique() #47 job titles
# The 47 jobs can be separated according to corporate hierarchy
# Use employee, manager, and combined executives and board 
employee = ['Meat Cutter', 'Dairy Person', 'Produce Clerk', 'Baker', 'Cashier',
            'Shelf Stocker', 'Recruiter', 'HRIS Analyst', 'Accounting Clerk',
            'Benefits Admin', 'Labor Relations Analyst', 'Accounts Receiveable Clerk',
            'Accounts Payable Clerk', 'Auditor', 'Compensation Analyst',
            'Investment Analyst', 'Systems Analyst', 'Corporate Lawyer', 'Legal Counsel']

manager = ['Customer Service Manager', 'Processed Foods Manager', 'Meats Manager',
           'Bakery Manager', 'Produce Manager', 'Store Manager', 'Trainer', 'Dairy Manager']

executive = ['Exec Assistant, Finance', 'Exec Assistant, Legal Counsel',
             'CHief Information Officer', 'CEO', 'Exec Assistant, Human Resources',
             'Exec Assistant, VP Stores']

board = ['VP Stores', 'Director, Recruitment', 'VP Human Resources', 'VP Finance',
         'Director, Accounts Receivable', 'Director, Accounting',
         'Director, Employee Records', 'Director, Accounts Payable',
         'Director, HR Technology', 'Director, Investments',
         'Director, Labor Relations', 'Director, Audit', 'Director, Training',
         'Director, Compensation']
# Check all jobs were entered into the categories
total = len(employee) + len(manager) + len(executive) + len(board)
print('Total jobs categorised:', total, 'out of 47')

# Make a copy of job titles in a new column
dataset['Hierarchy'] = dataset.job_title
# Replace the job titles in Hierarchy
# The corporate hierarchy intrinsically has order from small to large, 
# so ordinal numbers may be used
dataset.Hierarchy = dataset.Hierarchy.replace(employee, 0)
dataset.Hierarchy = dataset.Hierarchy.replace(manager, 1)
dataset.Hierarchy = dataset.Hierarchy.replace(executive, 2)
dataset.Hierarchy = dataset.Hierarchy.replace(board, 3)
#Check that the replacement went to plan
dataset.Hierarchy.value_counts()
dataset.Hierarchy.shape

# =============================================================================
# Create new categories for department names
# =============================================================================
# Look at full list of departments and frequency
dataset.department_name.value_counts()
dataset.department_name.nunique() #21 department names
# The departments can be separated according to whether they serve the customer
# or the business
serve_cus = ['Meats', 'Dairy', 'Produce', 'Bakery', 'Customer Service', 'Processed Foods']

serve_biz = ['Store Management', 'Executive', 'Recruitment', 'HR Technology',
             'Accounting', 'Employee Records', 'Accounts Receiveable',
             'Accounts Payable', 'Labor Relations', 'Training', 'Compensation',
             'Audit', 'Investment', 'Information Technology', 'Legal']
# Check all departments were entered into the categories
total = len(serve_cus) + len(serve_biz)
print('Total departments categorised:', total, 'out of 21')
# Make a copy of department names in a new column
dataset['Service_to'] = dataset.department_name
# Replace the department names in Service_to
dataset.Service_to = dataset.Service_to.replace(serve_cus, 'Customer')
dataset.Service_to = dataset.Service_to.replace(serve_biz, 'Business')

# Check the replacement went to plan
dataset.Service_to.value_counts()
dataset.Service_to.shape
# =============================================================================
# Create new categories for city names
# =============================================================================
# Look at full list of cities and frequency
dataset.city_name.value_counts()
dataset.city_name.nunique() #40 city names
# The cities are in Canada.
# The cities can be separated according to population size.

# The population data for 2011 was obtained from Statistics Canada
# http://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/index.cfm?Lang=E
# Used 2011 as it is the most recent before the last year of this dataset (2015)
city_pop_2011 = {'Vancouver':2313328,
                 'Victoria':344615,
                 'Nanaimo':146574,
                 'New Westminster':65976,
                 'Kelowna':179839,
                 'Burnaby':223218,
                 'Kamloops':85678,
                 'Prince George':71974,
                 'Cranbrook':19319,
                 'Surrey':468251,
                 'Richmond':190473,
                 'Terrace':11486,
                 'Chilliwack':77936,
                 'Trail':7681,
                 'Langley':25081,
                 'Vernon':38180,
                 'Squamish':17479,
                 'Quesnel':10007,
                 'Abbotsford':133497,
                 'North Vancouver':48196,
                 'Fort St John':18609,
                 'Williams Lake':10832,
                 'West Vancouver':42694,
                 'Port Coquitlam':55985,
                 'Aldergrove':12083,
                 'Fort Nelson':3561,
                 'Nelson':10230,
                 'New Westminister':65976,
                 'Grand Forks':3985,
                 'White Rock':19339,
                 'Haney':76052,
                 'Princeton':2724,
                 'Dawson Creek':11583,
                 'Bella Bella':1095,
                 'Ocean Falls':129,
                 'Pitt Meadows':17736,
                 'Cortes Island':1007,
                 'Valemount':1020,
                 'Dease Lake':58,
                 'Blue River':215}
# Population notes
# New Westminister is treated as a misspelling of New Westminster
# Used Haney in Maple Ridge, British Columbia because most of the other cities are in BC
# Used Bella Bella 1 (Indian reserve) for Bella Bella
# Used Central Coast A for Ocean Falls 
# Used Strathcona B for Cortes Island
# Used Dease Lake 9 (Indian reserve) for Dease Lake

# Check dictionary made correctly
print('Cities in dictionary:', len(city_pop_2011), 'out of 40')
# Make a copy of city names
dataset['Pop'] = dataset.city_name
# Map from city name to population
dataset.Pop = dataset.Pop.map(city_pop_2011)
# Make a new column for population category
dataset['Pop_category'] = dataset.Pop
# Categorise according to population size
# >= 100,000 is City
# 10,000 to 99,999 is Rural
# < 10,000 is Remote
# Guidance from Australian Institute of Health and Welfare
# http://www.aihw.gov.au/rural-health-rrma-classification/
city_ix = (dataset['Pop'] >= 100000)
rural_ix = ((dataset['Pop'] < 100000) & (dataset['Pop'] >= 10000))
remote_ix = (dataset['Pop'] < 10000)
dataset.loc[city_ix, 'Pop_category'] = 'City'
dataset.loc[rural_ix, 'Pop_category'] = 'Rural'
dataset.loc[remote_ix, 'Pop_category'] = 'Remote'
#Check the replacement went to plan
dataset.Pop_category.value_counts()

# As the category names are based on population size, the data could be represented
# by an ordinal category instead of a nominal category.
# Convert from nominal to ordinal 
dataset.Pop_category = dataset.Pop_category.replace('Remote', 0)
dataset.Pop_category = dataset.Pop_category.replace('Rural', 1)
dataset.Pop_category = dataset.Pop_category.replace('City', 2)
# Check the replacement went to plan
dataset.Pop_category.value_counts()
dataset.Pop_category.shape

# =============================================================================
# Convert STATUS from string to numerical
# =============================================================================
dataset.STATUS = dataset.STATUS.map({'ACTIVE':1, 'TERMINATED':0})
dataset.STATUS.value_counts()

# =============================================================================
# # Data visualisation
# =============================================================================
# Separate data to avoid one excessively overlapping the other in plots.
# One set for the terminated (out of company) and working (in company)
out_of_co = dataset[dataset.STATUS == 0]
in_co = dataset[dataset.STATUS == 1]

# Start with a broad look at each group in terms of age and length of service
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

ax1.scatter(out_of_co.age, out_of_co.length_of_service, color='r')
ax1.set_xlabel('Age')
ax1.set_ylabel('Length of service')
ax1.set_title('Out of company')

ax2.scatter(in_co.age, in_co.length_of_service, color='b')
ax2.set_xlabel('Age')
ax2.set_title('In company')
#Observations
#People may leave the company after working for any length of time from 0 to 25 years and any age from 20 to 60. Take a closer look at the distribution of ages and service times for terminations.

# Scatter plot of out of company dataset, with histograms of the axes
g = sns.jointplot(out_of_co.age, out_of_co.length_of_service, color='r')

#Observations
#There appear to be 3 peaks in the age when people stop working. There are 4 major peaks in the length of service before people stop working.
#The largest age peak of above 60 years old overlaps the service peak of 25 years. This would be people who are retiring from the work force.
#The second largest age peak of 20-25 years old overlaps the service peak of 0 years. This is likely people who are trying jobs to find something they would like.
#The third age peak of 29-34 overlaps the service peak of 8 years. This is likely people who have become tired of their work and want a career change. It may also be people who have family commitments that force them to change.
#The largest service peak of around 13 years overlaps with the age peak of above 60 years and with ages between 40-50. The group over 60 years old would be middle are people who changed careers to join the company.

# When someone leaves the company, look at their age, length of service, city size,
# and position in hierarchy. Separated by gender
g = sns.FacetGrid(out_of_co, col='Pop_category', row='Hierarchy', palette='Set1_r', 
                  hue='gender_short', margin_titles=True)
g = (g.map(plt.scatter, 'age', 'length_of_service').add_legend())

# Do the same for people who are working in the company
g = sns.FacetGrid(in_co, col='Pop_category', row='Hierarchy', palette='Set1_r', 
                  hue='gender_short', margin_titles=True)
g = (g.map(plt.scatter, 'age', 'length_of_service').add_legend())

#Observations
#Nobody has stopped employment when they were at executive level.
#Executives and board members only work in cities.
#There does not appear to be a major difference in employment termination between males and females.
#Managers and board members stop employment after at least 14 years of service. This means that they were likely internally promoted to those positions.

# Out of interest, look at when termination is voluntary or involuntary
g = sns.FacetGrid(out_of_co, col='Pop_category', row='termreason_desc', palette='Set1_r', 
                  hue='termtype_desc', margin_titles=True)
g = (g.map(plt.scatter, 'age', 'length_of_service').add_legend())

#Observations
#Layoffs occur for all ages and all service lengths in remote and rural areas.
#Resignations are uncommon in remote areas.
#As expected, layoffs are involuntary, whereas resignations and retirements are voluntary.

# Out of interest, look at number of terminations per year.
# Count terminations per year
from collections import Counter
term_per_year = Counter(out_of_co.STATUS_YEAR)
term_per_year_df = pd.DataFrame.from_dict(term_per_year, orient='index')
term_per_year_df = term_per_year_df.sort_index()
term_per_year_df.plot(kind='bar')

#Observations
#2014 had an unusually high number of employment terminations.
#There was a peak in employment terminations in 2007-2008, when the GFC occurred.
#There is another peak in 2012, but it is lower than the GFC peak.
#Employment terminations in 2015 is similar to the GFC peak, but much lower than the 2014 peak. The dataset reaches 31 December 2015, so the 2015 record is complete.

# =============================================================================
# Preprocessing for machine learning models
# =============================================================================
# Drop the employee ID, record date, birth date, termination date, termination reason, 
# termination type, gender_full, STATUS_YEAR, and store_name features
# Also drop job_title (replaced with Hierarchy), department_name (replaced with Service_to),
# and city_name and Pop (replaced with Pop_category)
drop_cols = ['EmployeeID', 'recorddate_key', 'birthdate_key', 'orighiredate_key',
             'terminationdate_key', 'gender_full', 'termreason_desc',
             'termtype_desc', 'STATUS_YEAR', 'store_name', 'job_title', 'department_name',
             'city_name', 'Pop']

dataset= dataset.drop(drop_cols, axis=1)
# The gender, business unit and Service_to categories are nominal, so they will
# be exploded instead of being converted to ordinal values
dummy_cols = ['gender_short', 'BUSINESS_UNIT', 'Service_to']
dataset= pd.get_dummies(dataset, columns=dummy_cols)

# Separate the label from the dataset
label = dataset.STATUS
dataset= dataset.drop('STATUS', axis=1)

# Machine learning classification models

# Only age and length of service have double digit values. It should be okay to
# leave the values at their normal scales.

# Split data for training and testing. Specify random state for repeatability.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3,
                                                    random_state=10)

# =============================================================================
# KNN
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('KNN model score is %0.4f' %score)

# =============================================================================
# SVC
# =============================================================================
from sklearn.svm import SVC
model = SVC(C=1, kernel='rbf', random_state=10)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('SVC model score is %0.4f' %score)

# =============================================================================
# Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2,
                               oob_score=False, random_state=10)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('Random Forest model score is %0.4f' %score)

'''Using default parameters for the K-Nearest Neighbours model, the SVC model, and 
the Random Forest model produce scores around 0.98, which is good.'''

