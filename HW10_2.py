import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering


_mondai = 3
mnist = np.load('mnist.npz')
X_tr, y_tr, _, _ = [mnist[i] for i in mnist.files]
X = []
y = []
for i in range(len(y_tr)):
    if y_tr[i][0] == 1 or y_tr[i][3] == 1 or y_tr[i][7] == 1:
        X.append(X_tr[i].flatten())
        if y_tr[i][0] == 1:
            y.append(0)
        elif y_tr[i][3] == 1:
            y.append(1)
        else:
            y.append(2)

if _mondai == 1:
    X, _, y, _ = train_test_split(X, y, train_size=1 / 4)  # train data sampling, rate = 0.25
    J_e = []
    SC = []
    for _K in range(1, 11):
        print('K = {} processing'.format(_K))
        model = KMeans(n_clusters=_K, init='random')
        model.fit(X)
        center = model.cluster_centers_
        label = model.labels_
        J_e_k = 0
        for i in range(len(X)):
            label_i = label[i]
            x_center = center[label_i]
            J_e_k += np.sum((X[i]-x_center)**2)
        J_e.append(J_e_k)
        if _K != 1:
            SC.append((silhouette_score(X, label)))
    plt.plot(J_e)
    plt.title('$J_e$ curve')
    plt.show()
    print('J_e = ', J_e)
    print('SC (2~10) = ', SC)
elif _mondai == 2:
    NMI = []
    _itertime = 10
    for i in range(_itertime):
        model = KMeans(n_clusters=3, init='random')
        model.fit(X)
        NMI_i = normalized_mutual_info_score(model.labels_, y)
        NMI.append(NMI_i)
    print('iter = {}, NMI = {}'.format(_itertime, np.mean(NMI)))
else:  # 一致聚类
    _itertime = 200
    model = KMeans(n_clusters=3, init='random')
    X, _, y, _ = train_test_split(X, y, train_size=1/10)
    print('set scale: ', len(X))
    X = np.array(X)
    full_sample_index = np.random.rand(_itertime*len(X))
    sum_M_ks = np.zeros((len(X), len(X)))
    sum_I_s = np.zeros((len(X), len(X)))
    s = 0
    for _iter in range(_itertime):
        sample_rate = 0.2
        sample_index = full_sample_index[len(X)*_iter: len(X)*(_iter+1)] < sample_rate
        #  sample_index 是采样用的定位集合，长度为n
        #  sample_index_num 是从全数据集定位符定位到采样样本集的转移矩阵，长度为n
        sample_index_num = np.zeros(len(sample_index))
        j = 1
        for i in range(len(sample_index)):
            if sample_index[i] == 1:
                sample_index_num[i] = j
                j += 1
        sample = X[sample_index]
        # sample为采样样本集，长度为m < n
        print('processed :{} , sample scale'.format(_iter), len(sample))
        label = model.fit_predict(sample)  # 采样并获取标签
        # 更新 I_s和M_ks矩阵
        for i in range(len(X)):
            if sample_index[i]:
                for j in range(len(X)):
                    if sample_index[j]:  # i, j均被采样，判断其是否在同一个分类中
                        sum_I_s[i][j] = sum_I_s[i][j] + 1
                        if label[int(sample_index_num[i])-1] == label[int(sample_index_num[j])-1]:
                            sum_M_ks[i][j] = sum_M_ks[i][j] + 1
    M_k = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            if sum_I_s[i, j] == 0:
                M_k[i, j] = 0
            else:
                M_k[i, j] = sum_M_ks[i, j]/sum_I_s[i, j]
    hierarchy_model = AgglomerativeClustering(n_clusters=3, linkage='average', affinity='precomputed')
    hierarchy_model.fit(1-M_k)

    result = np.zeros((len(X), len(X)))
    p = 0
    for i in range(3):
        for j in range(len(X)):
            if hierarchy_model.labels_[j] == i:
                result[:, p] = M_k[:, j]
                p += 1
    result1 = np.zeros((len(X), len(X)))
    p = 0
    for i in range(3):
        for j in range(len(X)):
            if hierarchy_model.labels_[j] == i:
                result1[p, :] = result[j, :]
                p += 1
    plt.matshow(result1)
    plt.title('$M^k$ cluster visualization')
    plt.show()
    NMI = normalized_mutual_info_score(hierarchy_model.labels_, y)
    print('NMI = ', NMI)

