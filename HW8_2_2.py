import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from minepy import MINE
from sklearn.tree import DecisionTreeClassifier


def within_class_scatter(samples, mean):
    s_in = 0
    for j in range(len(samples)):
        s_in += (samples[j] - mean) ** 2
    return s_in / len(samples)


originX = np.loadtxt('feature_selection_X.txt')
originY = np.loadtxt('feature_selection_Y.txt')
X_train, X_test, y_train, y_test = train_test_split(originX, originY, test_size=0.25)
model_noextract = LogisticRegression(solver='liblinear')
model_noextract.fit(X_train, y_train)
y_predict = model_noextract.predict(X_test)
print('No feature extract acc: {}'.format(np.sum(y_predict == y_test) / len(y_test)))
feature_num = [1, 5, 10, 20, 50, 100]
print('①argmax J = ln(|Sb|/|Sw|)')
Sb = np.zeros(1000)
Sw = np.zeros(1000)
X_train_0 = X_train[y_train == 0]
X_train_1 = X_train[y_train == 1]
P1 = np.sum(y_train) / len(y_train)
P0 = 1 - P1
for i in range(1000):
    '''对1000个特征分别计算类内类间距离比，选取最大的前n个特征'''
    group0 = X_train_0[:, i]
    group1 = X_train_1[:, i]
    m = np.mean(X_train[:, i])
    m0 = np.mean(group0)
    m1 = np.mean(group1)
    Sb[i] = P0 * (m0 - m) ** 2 + P1 * (m1 - m) ** 2
    Sw[i] = P0 * within_class_scatter(group0, m0) + P1 * within_class_scatter(group1, m1)
J = np.log(np.abs(Sb) / np.abs(Sw))
K = np.argsort(J)
feature1 = []
for i in feature_num:
    feature_position = K[1000 - i:1000]
    print('feature position:{}'.format(feature_position))
    feature1.append(feature_position)
    X_train_featured = X_train[:, feature_position]
    model_featured = LogisticRegression(solver='liblinear')
    model_featured.fit(X_train_featured, y_train)
    y_predict = model_featured.predict(X_test[:, feature_position])
    print('{} features acc: {}'.format(i, np.sum(y_predict == y_test) / len(y_test)))
print('②MINE')
mine = MINE()
J = np.zeros(1000)
for i in range(1000):
    mine.compute_score(X_train[:, i], y_train)
    J[i] = mine.mic()
K = np.argsort(J)
feature2 = []
for i in feature_num:
    feature_position = K[1000 - i:1000]
    print('feature position:{}'.format(feature_position))
    feature2.append(feature_position)
    X_train_featured = X_train[:, feature_position]
    model_featured = LogisticRegression(solver='liblinear')
    model_featured.fit(X_train_featured, y_train)
    y_predict = model_featured.predict(X_test[:, feature_position])
    print('{} features acc: {}'.format(i, np.sum(y_predict == y_test) / len(y_test)))
same_feature_num = []
for i in range(len(feature_num)):
    same_feature = 0
    for j in range(feature_num[i]):
        if feature1[i][j] in feature2[i]:
            same_feature += 1
    same_feature_num.append(same_feature)
print('①②Two Method has same features numbers:{}'.format(same_feature_num))

'''前向算法：迭代计算出最优分类。'''
# print('③Forward greedy')
# nfeature = 5
# SFS = SequentialFeatureSelector(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=nfeature,
#                                 direction='forward', n_jobs=8, scoring='accuracy')
# SFS.fit(X_train, y_train)
# feature3 = np.array(SFS.get_support(True))
# same_feature = 0
# same_feature2 = 0
# for j in range(nfeature):
#     if feature3[j] in feature1[3]:
#         same_feature += 1
#     if feature3[j] in feature2[3]:
#         same_feature2 += 1
# print('③Same features numbers with ①:{},with ②:{}'.format(same_feature, same_feature2))

'''决策树选择特征'''
print('④Decision Tree')
model_featured = DecisionTreeClassifier()
model_featured.fit(X_train, y_train)
J = model_featured.feature_importances_
zeronum = np.sum(J == 0)
K = np.argsort(J)
print(K[zeronum:1000])
