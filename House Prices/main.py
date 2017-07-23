import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

train_df = pd.read_csv("train.csv")

total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)

train_df.sort_values(by = 'GrLivArea', ascending = False)[:2]

train_df["SalePrice"] = np.log(train_df["SalePrice"])
train_df["GrLivArea"] = np.log(train_df["GrLivArea"])

train_df['HasBsmt'] = pd.Series(len(train_df['TotalBsmtSF']), index=train_df.index)
train_df['HasBsmt'] = 0 
train_df.loc[train_df['TotalBsmtSF']>0,'HasBsmt'] = 1

train_df.loc[train_df['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_df['TotalBsmtSF'])

train_df = pd.get_dummies(train_df)

cols = ["OverallQual","YearBuilt","TotalBsmtSF","FullBath","GrLivArea","GarageCars"]

X_train = train_df.drop("SalePrice",axis=1)[cols]
Y_train = train_df["SalePrice"]

lrclf = LinearRegression()

lrclf.fit(X_train,Y_train)

test_df = pd.read_csv("test.csv")

X_test = test_df.drop("Id",axis=1).copy()[cols]

X_test.loc[X_test["GarageCars"].isnull()]=0
X_test.loc[X_test["TotalBsmtSF"].isnull()]=0

Y_pred = lrclf.predict(X_test)

acc_lr = round(lrclf.score(X_train,Y_train)*100,2)

print(acc_lr)

submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": Y_pred
})

submission.to_csv('output.csv',index=False)