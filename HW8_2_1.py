import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
model1 = LinearRegression()
model2 = Ridge(alpha=1)
model3 = Lasso(alpha=1)
corrList = []
model1coef = []
model2coef = []
model3coef = []
for i in range(20):
    epsilon1 = np.random.uniform(0, 2, 20)
    epsilon2 = np.random.uniform(0, 0.5, 20)
    x1 = np.array(range(1, 21))
    y = 3*x1 + 2 + epsilon1
    x2 = 0.05*x1 + epsilon2
    X = np.vstack((x1, x2)).T
    corr = np.corrcoef(x1, x2)
    corrList.append(corr[0][1])
    model1.fit(X, y)
    model2.fit(X, y)
    model3.fit(X, y)
    model1coef.append(model1.coef_)
    model2coef.append(model2.coef_)
    model3coef.append(model3.coef_)
    print(model1.coef_, model2.coef_, model3.coef_)
print('corr mean:', np.mean(corrList))
print('adjust lambda:')
model1coef = np.array(model1coef)
model2coef = np.array(model2coef)
model3coef = np.array(model3coef)
plt.scatter(model1coef[:,0], model1coef[:,1], label='Linear')
plt.scatter(model2coef[:,0], model2coef[:,1], label='Ridge')
plt.scatter(model3coef[:,0], model3coef[:,1], label='LASSO')
plt.legend()
plt.show()
model2coef = []
model3coef = []
for _alpha in range(1, 21):
    model2 = Ridge(alpha=_alpha/2)
    model3 = Lasso(alpha=_alpha/2)
    model2.fit(X, y)
    model3.fit(X, y)
    model2coef.append(model2.coef_)
    model3coef.append(model3.coef_)
    print('lambda = {}'.format(_alpha/2), model2.coef_, sum(model2.coef_), model3.coef_)
model2coef = np.array(model2coef)
model3coef = np.array(model3coef)
plt.plot(np.array(range(1, 21))/2, model2coef, label='Ridge')
plt.plot(np.array(range(1, 21))/2, model3coef, label='LASSO')
plt.legend()
plt.show()