import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import datetime

ColName = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
Data = pd.read_csv('mammographic.txt', header=None, names=ColName)
Data = Data[~Data.isin(['?'])]
CleanData = Data.dropna()  # 考虑没有必要为了预测是否患病而去预测别的属性，直接删除缺失项
LRAcc = []
FRAcc = []
epoch = 1
for TrainIndex, TestIndex in KFold(n_splits=10).split(CleanData):
    TrainData = CleanData.iloc[TrainIndex].astype('int')
    TestData = CleanData.iloc[TestIndex].astype('int')
    # TODO: 将CleanDATA按照列做标准化
    X_Train = TrainData.loc[:, ('BI-RADS', 'Age', 'Shape', 'Margin', 'Density')]
    Y_Train = TrainData.loc[:, 'Severity']
    X_Test = TestData.loc[:, ('BI-RADS', 'Age', 'Shape', 'Margin', 'Density')]
    Y_Test = TestData.loc[:, 'Severity']
    Time_A = datetime.datetime.now()
    Model = LogisticRegression()
    Model.fit(X_Train, Y_Train)
    Y_Predict = Model.predict(X_Test)
    Time_B = datetime.datetime.now()
    Accuracy = 1 - sum(Y_Predict ^ Y_Test.values)/len(Y_Predict)
    LRAcc.append(Accuracy)
    print('Epoch {} \tLR Acc = {:.4f}'.format(epoch,Accuracy))
    Group0 = TrainData[~TrainData['Severity'].isin(['1'])].loc[:, ('BI-RADS', 'Age', 'Shape', 'Margin', 'Density')]
    Group1 = TrainData[~TrainData['Severity'].isin(['0'])].loc[:, ('BI-RADS', 'Age', 'Shape', 'Margin', 'Density')]
    Group0 = Group0.values.T
    Group1 = Group1.values.T  # 现在Group0和Group1都是列向量形式
    Time_C = datetime.datetime.now()
    Mean0 = np.mat(np.mean(Group0, axis=1)).T
    Mean1 = np.mat(np.mean(Group1, axis=1)).T

    def within_class_scatter(samples, mean):
        _Temp, nums = samples.shape[:2]
        samples_mean = samples.T - mean.T
        samples_mean = np.mat(samples_mean.T)
        s_in = np.zeros([5, 5])
        for j in range(nums):
            x = samples_mean[:, j]
            s_in += np.dot(x, x.T)
        return np.mat(s_in)
    S_in0 = within_class_scatter(Group0, Mean0)
    S_in1 = within_class_scatter(Group1, Mean1)
    S_w = S_in1 + S_in0
    S_wI = S_w.I
    w = np.dot(S_wI, Mean0 - Mean1)
    X_Test = TestData.loc[:, ('BI-RADS', 'Age', 'Shape', 'Margin', 'Density')]
    Y_Test = TestData.loc[:, 'Severity']
    X_Test = X_Test.values
    Y_Test = Y_Test.values
    k = 0
    for i in range(0, len(X_Test)):
        Sample = np.mat(X_Test[i]).T
        G = np.dot(w.T, Sample - 0.5*(Mean0 + Mean1))
        if (G >= 0) ^ Y_Test[i]:
            k += 1
    Time_D = datetime.datetime.now()
    FRAcc.append(k/len(Y_Test))
    print('Epoch {} \tFisher Acc = {:.4f}'.format(epoch, k/len(Y_Test)))
    print('\tTime Usage : LR = {}, LDA = {}'.format(Time_B-Time_A, Time_D-Time_C))
    epoch += 1
print('Logistic Regression Mean Acc = {:.4f}'.format(np.mean(LRAcc)))
print('Fisher Linear Mean Acc = {:.4f}'.format(np.mean(FRAcc)))
