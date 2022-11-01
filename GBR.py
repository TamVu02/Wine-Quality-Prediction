from sklearn import ensemble
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv('trainAlt.txt',";")
df = pd.DataFrame(datas)
X = df.iloc[ :, [0,1,2,3,4,5,6,7,8,9,10,12]].values
y = df.iloc[ :, 11].values

params = {'n_estimators': 2000,'max_depth': 3,'learning_rate': 0.1,'criterion': 'friedman_mse'}
gbrModel = ensemble.GradientBoostingRegressor(**params)
gbrModel.fit(X,y)

y_pred = gbrModel.predict(X)
y_test_copy = y
tup = []
for i in range(0,len(y)):
   tup.append((y_pred[i],y_test_copy[i]))
tup.sort(key = lambda x: x[0])
y0 = [tup[i][0] for i in range(0,len(y))]
y1 = [tup[i][1] for i in range(0,len(y))]

c = [i for i in range (1,len(y)+1,1)]
plt.scatter(c,y1,color='white',edgecolor = 'red',label="Actual Output")
plt.plot(c,y0,color='blue',label="Predicted Output",linestyle='-')
plt.xlabel('index')
plt.ylabel('charges')
plt.title('Prediction of GBR')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

print(gbrModel.score(X,y))

testData = pd.read_csv('test.csv',";")
df2 = pd.DataFrame(testData)
Xtest = df2.iloc[ :, 1:].values
id = df2.iloc[ :, 0].values
predict=gbrModel.predict(Xtest)

#write to file
result={'id':id,'quality':predict}
df3 = pd.DataFrame(result)
print('\n',df3)
df3.to_csv('result4.csv',index=False)
