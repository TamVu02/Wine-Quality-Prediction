from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

datas = pd.read_csv('trainAlt.txt',";")
df = pd.DataFrame(datas)

X = df.iloc[ :, [0,1,2,3,4,5,6,7,8,9,10,12]].values
y = df.iloc[ :, 11].values

X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
GradientBoostingRegressor(random_state=0)

print(reg.predict(X_test))
print(reg.score(X_test, y_test))
