import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

#read data
df = pd.read_csv('trainAlt.txt',";")
for col in df.columns:
    print('Unique value count of', col, 'is', len(df[col].unique()))
print('\n',df.head())

#slit data
X = df.loc[:, df.columns != 'quality']
Y = df['quality']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

adaboost = AdaBoostClassifier(n_estimators = 50, learning_rate = 0.2).fit(X_train, Y_train)
score = adaboost.score(X_test, Y_test)
print('Adaboost score: ',score)

# xgboost = XGBClassifier(n_estimators = 1000, learning_rate = 0.05).fit(X_train, Y_train, early_stopping_rounds = 5, eval_set = [(X_test, Y_test)],verbose = False)
# score_xgb = xgboost.score(X_test,Y_test)
# print('XGB score: ',score_xgb)

Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1)
predictions = np.mean(cross_validate(Tree_model,X,Y,cv=200)['test_score'])
print('The accuracy is: ',predictions*100,'%')


