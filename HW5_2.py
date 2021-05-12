import numpy as np
import os
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances as distance
import gc

_ClassNum = 10
_Method = 'XGBoost'  # Method = 'kNN' or 'cNN' or 'XGBoost'

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
if _Method == 'kNN':
    _k = 1
    KNN = KNeighborsClassifier(n_neighbors=_k, metric='cosine')
    KNN.fit(X_Train, Y_Train)
    Y_Predict = KNN.predict(X_Test)
    print(Y_Predict)
elif _Method == 'cNN':  # condensed NN
    def predict(TrainX, TrainY, Sample):
        Distance = np.zeros(len(TrainX))
        for k in range(0, len(TrainX)):
            Distance[k] = distance([TrainX[k]], [Sample])
        k = np.argmin(Distance)
        return TrainY[k]
    XS_X = []
    XS_Y = []
    XG_X = X_Train.tolist()
    XG_Y = Y_Train.tolist()
    XS_X.append(XG_X[0])
    XS_Y.append(XG_Y[0])
    del XG_Y[0]
    del XG_X[0]
    for i in range(0, len(XG_X)):
        if predict(XS_X, XS_Y, XG_X[i]) != XG_Y[i]:
            XS_X.append(XG_X[i])
            XS_Y.append(XG_Y[i])
            print(i)
    Test_X = X_Test.tolist()
    Test_Y = []
    for i in range(0, len(Test_X)):
        Test_Y.append(predict(XS_X, XS_Y, Test_X[i]))
    Y_Predict = np.array(Test_Y)
elif _Method == 'XGBoost':
    XGB = XGBClassifier(use_label_encoder=False, n_estimators=50, max_depth=4, subsample=0.6, colsample_bytree=0.9)
    XGB.fit(X_Train, Y_Train)
    Y_Predict = XGB.predict(X_Test)
    print(Y_Predict)

print('{} class(Method: {}) Acc:{}'.format(_ClassNum, _Method, sum(Y_Test == Y_Predict)/len(Y_Test)))
