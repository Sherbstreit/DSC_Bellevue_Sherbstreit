# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.features import Rank2D
import os
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC

desired_width = 300
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 12)

# set wd
os.chdir('/Users/saraherbstreit/Documents/week-6')

# import df and view headers
df = pd.read_csv("train.csv")
print(df.head())

# show dimensions
print("The dimension of the table is: ", df.shape)

# visualize data descriptions and summary
print("Describe Data")
print(df.describe())
print("Summarized Data\n")
print(df.describe(include=['O']))

# set figure size
plt.rcParams['figure.figsize'] = (20, 10)
# create subplots
fig, axes = plt.subplots(nrows=2, ncols=2)

# define x and y axis features
num_features = ['Age', 'SibSp', 'Parch', 'Fare']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts']

# return flattened array, draw histogram
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(df[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
plt.savefig('titanic_fig1.png')

# set figure size
plt.rcParams['figure.figsize'] = (20, 10)
# make subplots
fig, axes = plt.subplots(nrows=2, ncols=2)

# convert binary values to yes/no responses
X_Survived = df.replace(
    {'Survived': {1: 'yes', 0: 'no'}}
).groupby('Survived').size().reset_index(name='Counts')['Survived']
Y_Survived = df.replace(
    {'Survived': {1: 'yes', 0: 'no'}}
).groupby('Survived').size().reset_index(name='Counts')['Counts']

# place survived chart top left in subplot
axes[0, 0].bar(X_Survived, Y_Survived)
axes[0, 0].set_title('Survived', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# create categories from numbers
X_Pclass = df.replace(
    {'Pclass': {1: '1st', 2: '2nd', 3: '3rd'}}
).groupby('Pclass').size().reset_index(name='Counts')['Pclass']
Y_Pclass = df.replace(
    {'Pclass': {1: '1st', 2: '2nd', 3: '3rd'}}
).groupby('Pclass').size().reset_index(name='Counts')['Counts']

# place class chart in top right in subplot
axes[0, 1].bar(X_Pclass, Y_Pclass)
axes[0, 1].set_title('Pclass', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# Group data by sex
X_Sex = df.groupby('Sex').size().reset_index(
    name='Counts')['Sex']
Y_Sex = df.groupby('Sex').size().reset_index(
    name='Counts')['Counts']

# place sex chart on bottom left subplot
axes[1, 0].bar(X_Sex, Y_Sex)
axes[1, 0].set_title('Sex', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# group by embarked responses
X_Embarked = df.groupby('Embarked').size().reset_index(
    name='Counts')['Embarked']
Y_Embarked = df.groupby('Embarked').size().reset_index(
    name='Counts')['Counts']

# place embarked chart on bottom right subplot
axes[1, 1].bar(X_Embarked, Y_Embarked)
axes[1, 1].set_title('Embarked', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)
plt.savefig('titanic_fig2.png')

# set figure size
plt.rcParams['figure.figsize'] = (15, 7)
# convert features to numpy array
X = df[num_features].to_numpy()

# set up visualizer and define arguments
visualizer = Rank2D(features=num_features, algorithm='pearson')

# fit features to visualizer
visualizer.fit(X)
visualizer.transform(X)
# create png file of the image and also show in shell
visualizer.show(outpath="data1.png")
plt.savefig('titanic_fig3.png')

# set figure size and font size
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 50

# add pretty coloring scheme to graph
from yellowbrick.style import set_palette

set_palette('sns_bright')

# define classes and features to use
classes = ['Not-survived', 'Survived']
num_features = ['Age', 'SibSp', 'Parch', 'Fare']

# create a copy of original df
data_norm = df.copy()

# normalize values in new copy
for feature in num_features:
    data_norm[feature] = (df[feature] - df[feature].min(
        skipna=True)) / (df[feature].max(skipna=True) - df[feature].min(skipna=True))

# convert values to numpy arrays
X = data_norm[num_features].to_numpy()
y = df.Survived.to_numpy()

# set up visualizer
from yellowbrick.features import ParallelCoordinates

visualizer = ParallelCoordinates(classes=classes, features=num_features)

# fit visualizer
visualizer.fit(X, y)
visualizer.transform(X)
# create PNG file and also display in shell
visualizer.show(outpath="titanic_fig4.png")
visualizer.show()

# set figure size, make subplots
plt.rcParams['figure.figsize'] = (20, 10)
fig, axes = plt.subplots(nrows=2, ncols=2)

# convert binary to survived/not survived, group by sex
Sex_survived = df.replace(
    {'Survived': {1: 'Survived', 0: 'Not-survived'}}
)[df['Survived'] == 1]['Sex'].value_counts()
Sex_not_survived = df.replace(
    {'Survived': {1: 'Survived', 0: 'Not-survived'}}
)[df['Survived'] == 0]['Sex'].value_counts()
Sex_not_survived = Sex_not_survived.reindex(
    index=Sex_survived.index)

# place survived by sex chart on top left subplot
p1 = axes[0, 0].bar(Sex_survived.index, Sex_survived.values)
p2 = axes[0, 0].bar(Sex_not_survived.index, Sex_not_survived.values,
                    bottom=Sex_survived.values)
axes[0, 0].set_title('Sex', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)
axes[0, 0].legend((p1[0], p2[0]), ('Survived', 'Not-survived'), fontsize=15)

# convert binary to survived/not survived, group by class
Pclass_survived = df.replace({'Survived': {1: 'Survived', 0: 'Not-survived'}}
                             ).replace({'Pclass': {1: '1st', 2: '2nd', 3: '3rd'}}
                                       )[df['Survived'] == 1]['Pclass'].value_counts()
Pclass_not_survived = df.replace({'Survived': {1: 'Survived', 0: 'Not-survived'}}
                                 ).replace({'Pclass': {1: '1st', 2: '2nd', 3: '3rd'}}
                                           )[df['Survived'] == 0]['Pclass'].value_counts()
Pclass_not_survived = Pclass_not_survived.reindex(index=Pclass_survived.index)

# place survived by class chart on top right subplot
p3 = axes[0, 1].bar(Pclass_survived.index, Pclass_survived.values)
p4 = axes[0, 1].bar(Pclass_not_survived.index, Pclass_not_survived.values,
                    bottom=Pclass_survived.values)
axes[0, 1].set_title('Pclass', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)
axes[0, 1].legend((p3[0], p4[0]), ('Survived', 'Not-survived'), fontsize=15)

# convert binary to survived/not survived, group by embark response
Embarked_survived = df.replace({'Survived': {1: 'Survived', 0: 'Not-survived'}}
                               )[df['Survived'] == 1]['Embarked'].value_counts()
Embarked_not_survived = df.replace({'Survived': {1: 'Survived', 0: 'Not-survived'}}
                                   )[df['Survived'] == 0]['Embarked'].value_counts()
Embarked_not_survived = Embarked_not_survived.reindex(index=Embarked_survived.index)

# place survived by embark response chart on bottom left subplot
p5 = axes[1, 0].bar(Embarked_survived.index, Embarked_survived.values)
p6 = axes[1, 0].bar(Embarked_not_survived.index, Embarked_not_survived.values,
                    bottom=Embarked_survived.values)
axes[1, 0].set_title('Embarked', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)
axes[1, 0].legend((p5[0], p6[0]), ('Survived', 'Not-survived'), fontsize=15)
plt.savefig('titanic_fig5.png')


# replace NaN values with median
def fill_na_median(data, inplace=True):
    return data.fillna(data.median(), inplace=inplace)


# apply median function to Age column
fill_na_median(df['Age'])

# check the result
print(df['Age'].describe())


# replace NaN values with mode
def fill_na_most(data, inplace=True):
    return data.fillna('S', inplace=inplace)


# apply mode function to embarked column
fill_na_most(df['Embarked'])

# check the result
print(df['Embarked'].describe())


# log-transformation function
def log_transformation(data):
    return data.apply(np.log1p)


# create new variable with log transformation applied to Fare column
df['Fare_log1p'] = log_transformation(df['Fare'])

# check the data to see changes
print(df.describe())

# set up the figure size
plt.rcParams['figure.figsize'] = (10, 5)

# Create histogram of fare distribution
plt.hist(df['Fare_log1p'], bins=40)
plt.xlabel('Fare_log1p', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.savefig('titanic_fig6.png')

# identify categorical features
cat_features = ['Pclass', 'Sex', 'Embarked']
df_cat = df[cat_features]
# create categories from Pclass numbers
df_cat = df_cat.replace({'Pclass': {1: '1st', 2: '2nd', 3: '3rd'}})

# One Hot Encoding
df_cat_dummies = pd.get_dummies(df_cat)
# check the data
print(df_cat_dummies.head(8))

# here we will combine the numerical features and the dummie features together
features_model = ['Age', 'SibSp', 'Parch', 'Fare_log1p']
data_model_X = pd.concat([df[features_model], df_cat_dummies], axis=1)

# create a whole target dataset that can be used for train and validation data splitting
data_model_y = df.replace({'Survived': {1: 'Survived', 0: 'Not_survived'}})['Survived']
# separate data into training and validation and check the details of the datasets
# import packages
from sklearn.model_selection import train_test_split

# split the data
X_train, X_val, y_train, y_val = train_test_split(data_model_X, data_model_y, test_size =0.3, random_state=11)

# number of samples in each set
print("No. of samples in training set: ", X_train.shape[0])
print("No. of samples in validation set:", X_val.shape[0])

# Survived and not-survived
print('\n')
print('No. of survived and not-survived in the training set:')
print(y_train.value_counts())

print('\n')
print('No. of survived and not-survived in the validation set:')
print(y_val.value_counts())


# Instantiate the classification model
model = LogisticRegression()

#The ConfusionMatrix visualizer taxes a model
classes = ['Not_survived','Survived']
cm = ConfusionMatrix(model, classes=classes, percent=False)

#Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(X_train, y_train)

#To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
#and then creates the confusion_matrix from scikit learn.
cm.score(X_val, y_val)

# change fontsize of the labels in the figure
for label in cm.ax.texts:
    label.set_size(20)

#How did we do?
cm.show(outpath="cmdata.png")

# Precision, Recall, and F1 Score
# set the size of the figure and the font size
#%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 20

# Instantiate the visualizer
visualizer = ClassificationReport(model, classes=classes)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_val, y_val)  # Evaluate the model on the test data
g = visualizer.poof()

# ROC and AUC
#Instantiate the visualizer
visualizer = ROCAUC(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_val, y_val)  # Evaluate the model on the test data
g = visualizer.poof()
