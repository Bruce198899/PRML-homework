import numpy as np
import os
from skimage import io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

_ClassNum = 10
_Method = 'MLP'  # Method = 'LR' or 'MLP' or 'SVM'

Data = []
Target = []
for _face_class in range(0, _ClassNum):
    ImageList = os.listdir('Pictures/'+str(_face_class))
    for _image_num in range(0, len(ImageList)):
        Image = io.imread('Pictures/'+str(_face_class)+'/'+ImageList[_image_num], as_gray=True).flatten()
        Data.append(Image)
        Target.append(_face_class)
Data = StandardScaler().fit_transform(np.array(Data))
Target = np.array(Target)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(Data, Target, test_size=0.25)
Y_Predict = []
# Model and Predict
if _Method == 'LR':
    if _ClassNum != 2:
        LR = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)
    else:
        LR = LogisticRegression(solver='liblinear')
    Model = LR.fit(X_Train, Y_Train)
    Y_Predict = LR.predict(X_Test)
elif _Method == 'MLP':
    _DefaultSetting = False
    _Layer = (100, )  # i-th element shows neurons number of i-th layer
    _Activation = 'relu'  # logistic\relu\tanh
    _Solver = 'adam'  # sgd\adam
    _LR = 0.01  # init learning rate
    if _DefaultSetting:
        MLP = MLPClassifier()
    else:
        MLP = MLPClassifier(hidden_layer_sizes=_Layer, activation=_Activation, solver=_Solver, learning_rate_init=_LR)
    Y_Train_OneHot = np.zeros([len(Y_Train), _ClassNum])
    for i in range(0, len(Y_Train)):
        Y_Train_OneHot[i][int(Y_Train[i])] = 1
    MLP.fit(X_Train, Y_Train_OneHot)
    Y_Predict_OneHot = MLP.predict(X_Test)
    Y_Predict = np.zeros(len(Y_Test))
    for i in range(0, len(Y_Test)):
        Y_Predict[i] = np.argmax(Y_Predict_OneHot[i])
elif _Method == 'SVM':
    _DefaultSetting = True
    _C = 1
    _Gamma = 0.001
    _Kernel = 'rbf'
    if _DefaultSetting:
        SVM = SVC()
    else:
        SVM = SVC(kernel=_Kernel, C=_C, gamma=_Gamma)
    SVM.fit(X_Train, Y_Train)
    Y_Predict = SVM.predict(X_Test)

# Predict Analysis
if _ClassNum == 2:
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
        (TP+TN)/L, TP/(TP+FN), FP/(FP+TN), TN/(TN+FP), FN/(FN+TP)))
    print('Sens:{:.3f}\tSpec:{:.3f}\tFDR:{:.3f}'.format(TP/(TP+FN), TN/(TN+FP), FP/(FP+TP)))
    FPR, TPR, TH = roc_curve(Y_Test, Y_Predict)
    print('ROC: as figured\tAUC:{:3f}'.format(auc(FPR, TPR)))
    plt.plot(FPR, TPR, '-o')
    plt.show()
else:
    print('{} class Acc:{}'.format(_ClassNum, sum(Y_Test == Y_Predict)/len(Y_Test)))
