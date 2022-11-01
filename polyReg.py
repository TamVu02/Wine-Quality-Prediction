import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#read data
datas = pd.read_csv('trainAlt.txt',";")
df = pd.DataFrame(datas)

#add more feature
freeSul = df["free sulfur dioxide"]
totalSul = df["total sulfur dioxide"]
datas['log free sulfur dioxide']=np.log(df['free sulfur dioxide'] )
datas['log total sulfur dioxide']=np.log(df['total sulfur dioxide'] )

#extract data and output
X = df.iloc[ :, [0,1,2,3,4,5,6,7,8,9,10,12,13,14]].values
y = df.iloc[ :, 11].values

#split train and validate set
train = pd.DataFrame(X)
test = pd.DataFrame(y)
# set aside 20% of train and test data for evaluation
X_train, X_test, y_train, y_test = train_test_split(train, test,
    test_size=0.2, shuffle = True, random_state = 8)
# Use the same function above for the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
    test_size=0.25, random_state= 8) # 0.25 x 0.8 = 0.2

polyreg = make_pipeline(PolynomialFeatures(2), LinearRegression(fit_intercept=False))
linreg = LinearRegression()
scoring = "neg_root_mean_squared_error"
polyscores = cross_validate(polyreg, X_train, y_train, scoring=scoring, return_estimator=True)
linscores = cross_validate(linreg, X_train, y_train, scoring=scoring, return_estimator=True)
print(linscores["test_score"].mean())
print(polyscores["test_score"].mean())
print(linscores["test_score"].mean() - polyscores["test_score"].mean())
print(linscores["estimator"][0].intercept_, linscores["estimator"][-1].coef_)
linreg.fit(X_train, y_train)
print("Test set RMSE:", mean_squared_error(y_test, linreg.predict(X_test), squared=False))
print("Mean validation RMSE:", -linscores["test_score"].mean())

# print(f'free sulfur dioxide range: {freeSul.min()} to {freeSul.max()}')
# print(f'total sulfur dioxide range: {totalSul.min()} to {totalSul.max()}')
# print(f'residual sugar range: {sugar.min()} to {sugar.max()}')

# df = df[df.quality != 2]
# df = df[df.quality != 9]

# gbr=GradientBoostingRegressor( loss = 'huber',learning_rate=0.07,n_estimators=350, max_depth=6,subsample=1,verbose=False)
# gbr.fit(X,y)
# GB_accuracies = cross_val_score(estimator = gbr, X = X, y = y, cv = 10)
# print("Mean_GB_Acc : ", GB_accuracies.mean())

# Fitting Polynomial Regression to the dataset
poly = PolynomialFeatures(degree = 2,include_bias=False)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

y_pred_poly = lin2.predict(poly.fit_transform(X))
y_test_copy = y
tup = []
for i in range(0,len(y)):
   tup.append((y_pred_poly[i],y_test_copy[i]))
tup.sort(key = lambda x: x[0])
y0 = [tup[i][0] for i in range(0,len(y))]
y1 = [tup[i][1] for i in range(0,len(y))]

c = [i for i in range (1,len(y)+1,1)]
plt.scatter(c,y1,color='white',edgecolor = 'red',label="Actual Output")
plt.plot(c,y0,color='blue',label="Predicted Output",linestyle='-')
plt.xlabel('index')
plt.ylabel('charges')
plt.title('Prediction of Polynomial Regression')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
print('MSE Accuracy: ',mean_squared_error(y_test_copy, y_pred_poly,squared=False))

#compute test data
testData = pd.read_csv('test.csv',";")
df2 = pd.DataFrame(testData)
testData['log free sulfur dioxide']=np.log(df2['free sulfur dioxide'] )
testData['log total sulfur dioxide']=np.log(df2['total sulfur dioxide'] )
Xtest = df2.iloc[ :, 1:].values
id = df2.iloc[ :, 0].values
predict=lin2.predict(poly.fit_transform(Xtest))

#write to file
result={'id':id,'quality':predict}
df3 = pd.DataFrame(result)
print('\n',df3)
df3.to_csv('result4.csv',index=False)





