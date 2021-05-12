import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

Train = pd.read_table("prostate_train.txt")
# Train['ab'] = Train['lbph']*Train['svi']
# print(Train)
X = Train.loc[:, ('lcavol', 'lweight', 'lbph', 'svi')]
# X = Train.loc[:, ('lcavol', 'lweight', 'lbph', 'svi','ab')]
Y = Train.loc[:, 'lpsa']
Model = LinearRegression()
Model = Model.fit(X, Y)
print('Coef：{}, Intercept：{}'.format(Model.coef_, Model.intercept_))
Y_hat_Train = Model.predict(X)
SSR = sum((Y - Y_hat_Train)**2)
SST = sum((Y - np.mean(Y))**2)
RSquared = 1 - (SSR/SST)
print('Train set： SSR = {}, R^2 = {}'.format(SSR, RSquared))
Test = pd.read_table("prostate_test.txt")
# Test['ab'] = Test['lbph']*Test['svi']
X_test = Test.loc[:, ('lcavol', 'lweight', 'lbph', 'svi')]
# X_test = Test.loc[:, ('lcavol', 'lweight', 'lbph', 'svi','ab')]
Y_hat_Test = Model.predict(X_test)
Y_test = Test.loc[:, 'lpsa']
SSR_test = sum((Y_test - Y_hat_Test)**2)
SST_test = sum((Y_test - np.mean(Y_test))**2)
RSquared_test = 1 - (SSR_test/SST_test)
print('Test set： SSR = {}, R^2 = {}'.format(SSR_test, RSquared_test))
