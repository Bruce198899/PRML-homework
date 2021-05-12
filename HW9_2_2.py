import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import matplotlib.pyplot as plt

_Method = 'PCA'  # 'TSNE' OR 'PCA'
_PCAdim = [1, 2, 10, 50, 200, 300]
mnist = np.load('mnist.npz')
X_train, y_train, X_test, y_test = [mnist[i] for i in mnist.files]
X_train_new = []
X_test_new = []
y_train_new = []
y_test_new = []
for i in range(len(y_train)):
    if y_train[i][0] == 1 or y_train[i][9] == 1:
        X_train_new.append(X_train[i].flatten())
        if y_train[i][0] == 1:
            y_train_new.append(0)
        else:
            y_train_new.append(1)
for i in range(len(y_test)):
    if y_test[i][0] == 1 or y_test[i][9] == 1:
        X_test_new.append(X_test[i].flatten())
        if y_test[i][0] == 1:
            y_test_new.append(0)
        else:
            y_test_new.append(1)
if _Method == 'PCA':
    pca = PCA(n_components=2)
    X_train_new_trans = pca.fit_transform(X_train_new)
    X = np.vstack((X_train_new, X_test_new))
    y = np.concatenate((np.array(y_train_new), np.array(y_test_new)))

    model = SVC()
    model.fit(X_train_new, y_train_new)
    Y_Predict = model.predict(X_test_new)
    print('Before PCA: ACC ={}'.format(sum(Y_Predict == y_test_new) / len(y_test_new)))
    for dim in _PCAdim:
        pca1 = PCA(n_components=dim)
        pca1.fit(X)
        model.fit(pca1.transform(X_train_new), y_train_new)
        Y_Predict = model.predict(pca1.transform(X_test_new))
        print('After PCA(dim={}): ACC ={}'.format(dim, sum(Y_Predict == y_test_new)/len(y_test_new)))
elif _Method == 'TSNE':
    tsne = TSNE()
    X_train_new_trans = tsne.fit_transform(X_train_new)
print('Process Finished')
group0 = []
group1 = []
for i in range(len(X_train_new_trans)):
    if y_train_new[i] == 0:
        group0.append(X_train_new_trans[i])
    else:
        group1.append(X_train_new_trans[i])
group0 = np.array(group0)
group1 = np.array(group1)
plt.figure(figsize=(8, 8))
plt.scatter(group0[:, 0], group0[:, 1], marker='+', label='0')
plt.scatter(group1[:, 0], group1[:, 1], marker='+', label='9')
plt.title(_Method)
plt.legend()
plt.show()
