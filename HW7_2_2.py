import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import roc_curve, auc

data = np.array(pd.read_csv('spambase.data', header=None))
x = data[:, 0:56]
y = np.array(data[:, 57])
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=1000/4601)
model = MultinomialNB()
# model = GaussianNB()
model.fit(X_Train, Y_Train)
Y_Predict = model.predict(X_Test)

T, F, P, N = sum(Y_Predict), len(Y_Predict) - sum(Y_Predict), sum(Y_Test), len(Y_Test) - sum(Y_Test)
TP, FP, FN, TN = 0, 0, 0, 0
for _iter in range(0, len(Y_Test)):
    if Y_Test[_iter] == 1 and Y_Predict[_iter] == 1:
        TP += 1
    elif Y_Test[_iter] == 1 and Y_Predict[_iter] == 0:
        FN += 1
    elif Y_Test[_iter] == 0 and Y_Predict[_iter] == 0:
        TN += 1
    elif Y_Test[_iter] == 0 and Y_Predict[_iter] == 1:
        FP += 1
print('TP:{}\tFP:{}\nFN:{}\tTN:{}'.format(TP, FP, FN, TN))
L = len(Y_Test)
print('ACC:{:.3f}\tTPR:{:.3f}\t FPR:{:.3f}\tTNR:{:.3f}\tFNR:{:.3f}\t'.format(
    (TP + TN) / L, TP / (TP + FN), FP / (FP + TN), TN / (TN + FP), FN / (FN + TP)))
print('Sens:{:.3f}\tSpec:{:.3f}\tFDR:{:.3f}'.format(TP / (TP + FN), TN / (TN + FP), FP / (FP + TP)))
FPR, TPR, TH = roc_curve(Y_Test, Y_Predict)
print('ROC: as figured\tAUC:{:3f}'.format(auc(FPR, TPR)))
plt.plot(FPR, TPR, '-o')
plt.show()