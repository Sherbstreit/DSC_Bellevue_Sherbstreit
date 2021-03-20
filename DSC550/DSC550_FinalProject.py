# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# Allow headers to be completely viewed
desired_width = 300
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

# set wd
os.chdir('/Users/saraherbstreit/Documents')

# import df and view dimension and headers
df = pd.read_csv('credit_risk_dataset.csv')
print('The dimension of the full dataframe is: ', df.shape)
print(df.head())

# shorten long variable names
df = df.rename(columns={'person_age': 'age', 'person_income': 'income',
                        'person_home_ownership': 'home_stat',
                        'person_emp_length': 'length_emp',
                        'loan_int_rate': 'int_rate', 'loan_percent_income': 'DTI',
                        'cb_person_default_on_file': 'past_default',
                        'cb_person_cred_hist_length': 'cred_hist'})

# remove erroneous values
df = df[df['age'] < 100]
df = df[df['length_emp'] < 70]

# Keep only data on personal loans
df = df[df['loan_intent'] == 'PERSONAL']
print('The dimension of the personal dataframe is: ', df.shape)

# check columns with null values
print('null value count by variable', df.isnull().sum(axis = 0))

# remove int rate column
df.drop('int_rate', axis=1, inplace=True)

# create variable for label encoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
df['home_stat_cat'] = labelencoder.fit_transform(df['home_stat'])
df['loan_intent_cat'] = labelencoder.fit_transform(df['loan_intent'])
df['loan_grade_cat'] = labelencoder.fit_transform(df['loan_grade'])
df['past_default_cat'] = labelencoder.fit_transform(df['past_default'])

# check new df variables
print(df.head())

# define features
X = df.iloc[:, [0, 1, 3, 6, 8, 10, 11, 13, 14]]
# define target
y = df.iloc[:, 7]

#apply SelectKBest class to view best features
bestfeatures = SelectKBest(score_func=chi2, k=9)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# create labels for feature name and score
featureScores.columns = ['Feature','Score']
#  display feature scores
print(featureScores.nlargest(9,'Score'))


## Build logistic regression model
# remove age and credit_hist features
X = df.iloc[:, [1, 3, 6, 8, 11, 13, 14]]
# scale data
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create logistic regression model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# check performance
print('Accuracy of logistic regression model: %.3f (%.3f)' % (mean(scores), std(scores)))


## build random forest model
variables = df[['income', 'loan_status','loan_amnt', 'home_stat_cat', 'loan_grade_cat',
            'length_emp', 'past_default_cat', 'DTI']]
# define labels
labels = np.array(variables['loan_status'])
# Remove the label from the features
features = variables.drop('loan_status', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                            labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)
# make predictions from test data
predictions = rf.predict(test_features)
# check accuracy
print('R^2 of Random Forest model %.3f' % r2_score(test_labels, predictions))
