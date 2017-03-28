import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from sklearn.datasets import load_boston
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from collections import defaultdict

import csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
train = pd.read_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/train.csv")
kaggle_test = pd.read_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/kaggle_test.csv")
zip = pd.read_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/neighbourhood_zips.csv")
train_1 = train.copy()

#train_1.LotFrontage.fillna(0)

#print train.dtypes

############################### Replacing Nulls #########################################################

#train_1['LotFrontage'].replace(nan, 0)

train_1.LotFrontage[train_1.LotFrontage.isnull()] = 0 #replacing nulls in LotFrontage
train_1.MasVnrArea[train_1.MasVnrArea.isnull()] = 0 #replacing nulls in MasVnrArea
train_1.GarageYrBlt[train_1.GarageYrBlt.isnull()] = 2020 #replacing nulls in GarageYrBlt
train_1.MasVnrType[train_1.MasVnrType.isnull()] = 'None' #replacing nulls in MasVnrType
train_1.Electrical[train_1.Electrical.isnull()] = train_1.Electrical.mode() #replacing nulls in Electrical with mode of data
for c in train_1:
  if train_1[c].dtype == object:
    train_1.loc[train_1[c].isnull(),c] = 'NA' #replacing nulls in all Categorical Variables

# print train_1.GarageType[train_1.GarageYrBlt.isnull()].unique() checking if GarageType is null where GarageYrBlt is null

# if isinstance(train, pd.DataFrame): # To ckeck if a variable is a dataframe
  # print "do stuff"

# lot = train.MasVnrArea.unique()
# lot.sort()
# print lot
# lot_1 = train_1.MasVnrArea.unique()
# lot_1.sort()
# print lot_1

# train_1.GarageType[train_1.GarageType.isnull()] = 'NA' 
# print train_1.Electrical.unique()
# print len(train_1.MasVnrType[train_1.MasVnrType == 'NA']) 

# for item in train_1['Neighborhood'].unique():
  # print item
# response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address=Bloomington+Heights,Ames')
# resp_json_payload = response.json()
# print resp_json_payload["results"][0]["address_components"][6]["short_name"]



train_2 = pd.merge(train_1, zip, left_on='Neighborhood', right_on='Neighbourhood', how = 'left')

# print train_2['Zip'].unique()


# print train_2.Neighborhood[train_2.Zip.isnull()]

# print train_2.columns.values 

# print train_1.Neighborhood.unique()


# for c in train_2:
  # if train_2[c].dtype == object:
    # print c, len(train_2[c].unique()) #count unique values in every column

# print train_2['LandContour'].unique()


# sns.set(style="whitegrid", color_codes=True)



# overall_quality = train_2.groupby(['OverallQual'])['SalePrice'].apply(np.median)
# overall_quality.name = 'm_OverallQual'
# train_3 = train_2.join(overall_quality, on=['OverallQual'])

# temp = train_3.groupby(['OverallCond'])['SalePrice'].apply(np.median)
# temp.name = 'm_OverallCond'
# train_3 = train_3.join(temp, on=['OverallCond'])
# temp = train_3.groupby(['ExterQual'])['SalePrice'].apply(np.median)
# temp.name = 'm_ExterQual'
# train_3 = train_3.join(temp, on=['ExterQual'])
# temp = train_3.groupby(['ExterCond'])['SalePrice'].apply(np.median)
# temp.name = 'm_ExterCond'
# train_3 = train_3.join(temp, on=['ExterCond'])
# temp = train_3.groupby(['BsmtQual'])['SalePrice'].apply(np.median)
# temp.name = 'm_BsmtQual'
# train_3 = train_3.join(temp, on=['BsmtQual'])
# temp = train_3.groupby(['BsmtCond'])['SalePrice'].apply(np.median)
# temp.name = 'm_BsmtCond'
# train_3 = train_3.join(temp, on=['BsmtCond'])
# temp = train_3.groupby(['BsmtExposure'])['SalePrice'].apply(np.median)
# temp.name = 'm_BsmtExposure'
# train_3 = train_3.join(temp, on=['BsmtExposure'])
# temp = train_3.groupby(['BsmtFinType1'])['SalePrice'].apply(np.median)
# temp.name = 'm_BsmtFinType1'
# train_3 = train_3.join(temp, on=['BsmtFinType1'])
# temp = train_3.groupby(['BsmtFinType2'])['SalePrice'].apply(np.median)
# temp.name = 'm_BsmtFinType2'
# train_3 = train_3.join(temp, on=['BsmtFinType2'])
# temp = train_3.groupby(['HeatingQC'])['SalePrice'].apply(np.median)
# temp.name = 'm_HeatingQC'
# train_3 = train_3.join(temp, on=['HeatingQC'])
# temp = train_3.groupby(['BsmtFullBath'])['SalePrice'].apply(np.median)
# temp.name = 'm_BsmtFullBath'
# train_3 = train_3.join(temp, on=['BsmtFullBath'])
# temp = train_3.groupby(['BsmtHalfBath'])['SalePrice'].apply(np.median)
# temp.name = 'm_BsmtHalfBath'
# train_3 = train_3.join(temp, on=['BsmtHalfBath'])
# temp = train_3.groupby(['FullBath'])['SalePrice'].apply(np.median)
# temp.name = 'm_FullBath'
# train_3 = train_3.join(temp, on=['FullBath'])
# temp = train_3.groupby(['HalfBath'])['SalePrice'].apply(np.median)
# temp.name = 'm_HalfBath'
# train_3 = train_3.join(temp, on=['HalfBath'])
# temp = train_3.groupby(['KitchenQual'])['SalePrice'].apply(np.median)
# temp.name = 'm_KitchenQual'
# train_3 = train_3.join(temp, on=['KitchenQual'])
# temp = train_3.groupby(['Functional'])['SalePrice'].apply(np.median)
# temp.name = 'm_Functional'
# train_3 = train_3.join(temp, on=['Functional'])
# temp = train_3.groupby(['FireplaceQu'])['SalePrice'].apply(np.median)
# temp.name = 'm_FireplaceQu'
# train_3 = train_3.join(temp, on=['FireplaceQu'])
# temp = train_3.groupby(['GarageQual'])['SalePrice'].apply(np.median)
# temp.name = 'm_GarageQual'
# train_3 = train_3.join(temp, on=['GarageQual'])
# temp = train_3.groupby(['GarageCond'])['SalePrice'].apply(np.median)
# temp.name = 'm_GarageCond'
# train_3 = train_3.join(temp, on=['GarageCond'])
# temp = train_3.groupby(['PavedDrive'])['SalePrice'].apply(np.median)
# temp.name = 'm_PavedDrive'
# train_3 = train_3.join(temp, on=['PavedDrive'])
# temp = train_3.groupby(['PoolQC'])['SalePrice'].apply(np.median)
# temp.name = 'm_PoolQC'
# train_3 = train_3.join(temp, on=['PoolQC'])
# temp = train_3.groupby(['Fence'])['SalePrice'].apply(np.median)
# temp.name = 'm_Fence'
# train_3 = train_3.join(temp, on=['Fence'])

# overall_quality = train_3.groupby(['LandContour'])['SalePrice'].apply(np.median)
# overall_quality.name = 'm_LandContour'
# train_3 = train_2.join(overall_quality, on=['LandContour'])

# sns.stripplot(x = "LandContour", y= "m_LandContour", data = train_3, jitter=True)
# sns.plt.show()

# sns.stripplot(x = "OverallQual", y= "m_OverallQual", data = train_3, jitter=True)
# sns.plt.show()

# sns.stripplot(x = "OverallCond", y= "m_OverallCond", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "ExterQual", y= "m_ExterQual", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "ExterCond", y= "m_ExterCond", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "BsmtQual", y= "m_BsmtQual", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "BsmtCond", y= "m_BsmtCond", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "BsmtExposure", y= "m_BsmtExposure", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "BsmtFinType1", y= "m_BsmtFinType1", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "BsmtFinType2", y= "m_BsmtFinType2", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "HeatingQC", y= "m_HeatingQC", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "BsmtFullBath", y= "m_BsmtFullBath", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "BsmtHalfBath", y= "m_BsmtHalfBath", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "FullBath", y= "m_FullBath", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "HalfBath", y= "m_HalfBath", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "KitchenQual", y= "m_KitchenQual", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "Functional", y= "m_Functional", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "FireplaceQu", y= "m_FireplaceQu", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "GarageQual", y= "m_GarageQual", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "GarageCond", y= "m_GarageCond", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "PavedDrive", y= "m_PavedDrive", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "PoolQC", y= "m_PoolQC", data = train_3, jitter=True)
# sns.plt.show()
# sns.stripplot(x = "Fence", y= "m_Fence", data = train_3, jitter=True)
# sns.plt.show()

##############Converting ordinal variables into numeric ###########################



# train_2=pd.train_2({'ExterQual':['Ex', 'Gd', 'TA', 'Fa', 'Po']})
# conv_dict={'Ex':5.,'Gd':4.,'TA':3.,'Fa':2.,'Po':1.}
# train_2['ExterQual_c']=train_2.ExterQual.apply(conv_dict.get)

print "\n\n"
train_2['ExterQual'] = train_2['ExterQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})   
train_2['ExterCond'] = train_2['ExterCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})   
train_2['BsmtQual'] = train_2['BsmtQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})   
train_2['BsmtCond'] = train_2['BsmtCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})   
train_2['BsmtExposure'] = train_2['BsmtExposure'].map({'Gd':5, 'Av':4, 'Mn':3, 'No':1, 'NA':0})   
train_2['HeatingQC'] = train_2['HeatingQC'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
train_2['KitchenQual'] = train_2['KitchenQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})   
train_2['FireplaceQu'] = train_2['FireplaceQu'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})   
train_2['GarageQual'] = train_2['GarageQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})   
train_2['GarageCond'] = train_2['GarageCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})   
train_2['PoolQC'] = train_2['PoolQC'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})   
# w['female'] = w['female'].map({'female': 1, 'male': 0})

# print "\n\n"

# print train_2.ExterQual.unique()
# print "\n"
# print train_2.ExterCond.unique()
# print "\n"
# print train_2.BsmtQual.unique()
# print "\n"
# print train_2.BsmtCond.unique()
# print "\n"
# print train_2.BsmtExposure.unique()
# print "\n"
# print train_2.HeatingQC.unique()
# print "\n"
# print train_2.KitchenQual.unique()
# print "\n"
# print train_2.FireplaceQu.unique()
# print "\n"
# print train_2.GarageQual.unique()
# print "\n"
# print train_2.GarageCond.unique()
# print "\n"
# print train_2.PoolQC.unique()





train_3 = train_2.copy()

del train_3['Id']
del train_3['Neighborhood']
del train_3['MiscFeature']
del train_3['Neighbourhood']
Target = train_3['SalePrice'].copy()
del train_3['SalePrice']

##Converting categorical variables to numeric

train_3= pd.get_dummies(train_3)

# train_3.to_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/after_categorical_treatment.csv", sep=',')

#### Decision Trees for feature reduction

# X = train_3
# Y = Target
# names = train_3["feature_names"]
# rf = RandomForestRegressor()
# rf.fit(X, Y)
# print "Features sorted by their score:"
# print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True)

# regressor = RandomForestRegressor(n_estimators=150, min_samples_split=2)
# regressor.fit(train_3, Target)
# print train_3
# print regressor.predict(train_3)

X = train_3
Y = Target

# ShuffleSplit(Total number of records, n_iter= number of iterations , test_size = fraction or number of rows to give in test data, rest of the rows or fraction will go to training data):

rf = RandomForestRegressor()
scores = defaultdict(list)
names = list(X.columns.values)

# crossvalidate the scores on a number of different random splits of the data
for (train_idx, test_idx) in ShuffleSplit(len(X),n_iter=100,test_size = 0.2):
  temp = []
  X_train, X_test = X.loc[train_idx], X.loc[test_idx]
  Y_train, Y_test = Y.loc[train_idx], Y.loc[test_idx]
  r = rf.fit(X_train, Y_train)
  acc = r2_score(Y_test, rf.predict(X_test))

  for i in range(X.shape[1]):
    X_t = X_test.copy()
    X_t.iloc[:,i] = np.random.permutation(X_t.iloc[:,i].tolist())
    shuff_acc = r2_score(Y_test, rf.predict(X_t))
    scores[names[i]].append(abs(acc-shuff_acc)/acc)
sorted_scores = sorted([(feat, round(np.mean(score), 4) ) for feat, score in scores.items()], reverse=True, key=itemgetter(1))
# print sorted_scores
# for i in sorted_scores:
  # print type(i)
  # print i

# sorted_scores = sorted(sorted_scores, key=itemgetter(1))

with open('C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/scores.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for i in sorted_scores:
       writer.writerow(i)

reduced_features = []
for i in range(20):
  reduced_features.append(sorted_scores[i][0])

  # reduced_features = sorted_scores[0:19][0]       
# print reduced_features

  
X_reduced = X.loc[:,reduced_features]
# print X_reduced
  
X_reduced.to_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/train_reduced.csv", sep=',')

num_folds = 10
num_instances = len(X_reduced)
seed = 7
num_trees = 3

# params = {'n_estimators': 500, 'max_depth': 6,'learning_rate': 0.1, 'loss': 'lf','alpha':0.95}
# kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
# model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
# results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring = 'r2')

# print(results.mean())


for (train_idx, test_idx) in ShuffleSplit(len(X_reduced),n_iter=1,test_size = 0.2):
  X_train_r, X_test_r = X.loc[train_idx], X.loc[test_idx]
  Y_train_r, Y_test_r = Y.loc[train_idx], Y.loc[test_idx]

Y_test_r.to_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/Y_test_r.csv", sep=',')
Y_train_r.to_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/Y_train_r.csv", sep=',')
# print Y_train_r.tolist()
# for i in Y_train_r:
  # print type(i)

params = {'n_estimators': 100, 'max_depth': 10,'learning_rate': 0.1, 'loss': 'ls'}
clf = GradientBoostingRegressor(**params).fit(X_train_r, Y_train_r.tolist())
mse = mean_squared_error(Y_test_r.tolist(), clf.predict(X_test_r))

print "below is mse\n"
print mse
# print "below is the model prediction"
# print Y_test_r.tolist(), clf.predict(X_test_r)
# for i in Y_test_r:
  # print i
# for i in clf.predict(X_test_r):
  # print i

with open('C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/Actuals.csv', 'wb') as csv_file:
  writer = csv.writer(csv_file)
  for i in Y_test_r:
    writer.writerow([i])
    
with open('C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/Predicted.csv', 'wb') as csv_file:
  writer = csv.writer(csv_file)
  for i in clf.predict(X_test_r):
    writer.writerow([i])



kaggle_test_reduced = kaggle_test.loc[:,reduced_features]
kaggle_test_reduced.to_csv("C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/kaggle_test_reduced.csv", sep=',')

with open('C:/Users/M MANASA/Desktop/courses/Data Mining/Data Mining Housing Project/kaggle_test_predicted.csv', 'wb') as csv_file:
  writer = csv.writer(csv_file)
  for i in clf.predict(kaggle_test_reduced):
    writer.writerow([i])




