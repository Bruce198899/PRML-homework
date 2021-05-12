import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def parzen(value, target, h=1):
    Y = []
    for i in range(0, len(value)):
        prob = 0
        for j in range(0, len(target)):
            prob += np.exp(-(value[i] - target[j])*(value[i] - target[j])/(2*h*h)) / (h*np.sqrt(2*np.pi))
        Y.append(prob/len(target))
    return Y


Class0 = np.random.normal(-2.5, 1, 250)
Class1 = np.random.normal(2.5, 2, 250)
#  0类为正样本， 1类为负样本
Train0, Test0 = train_test_split(Class0, test_size=0.3)
Train1, Test1 = train_test_split(Class1, test_size=0.3)  # 这里没有赋标签是因为没有必要
X = np.linspace(-10, 10, 401)
plt.plot(X, parzen(X, Train0, 1))
plt.plot(X, parzen(X, Train1, 1))
plt.show()
# 没有给标签，所以0、1测试集要分开测试，结果和给标签是一样的
correct = 0
correctR = 0
Prob00 = parzen(Test0, Train0)
Prob01 = parzen(Test1, Train1)  # 描述了把0类分为1类的概率
for i in range(len(Test0)):
    if Prob00[i]/Prob01[i] >= 1:
        correct += 1
    R0 = Prob00[i]*0 + Prob01[i]*1  # 把x分为0类的代价 = 实际是0的后验概率*（预测0实际0）代价 + 实际是1的后验概率*（预测0实际1）代价
    R1 = Prob00[i]*10 + Prob01[i]*0
    if R0 <= R1:
        correctR += 1
Prob10 = parzen(Test1, Train0)
Prob11 = parzen(Test1, Train1)
for i in range(len(Test1)):
    if Prob11[i]/Prob10[i] >= 1:
        correct += 1
    R0 = Prob10[i] * 0 + Prob11[i] * 1
    R1 = Prob10[i] * 10 + Prob11[i] * 0
    if R1 <= R0:
        correctR += 1
print('最小错误率贝叶斯决策准确率：', correct/(len(Test0)+len(Test1)))
print('最小损失贝叶斯决策准确率：', correctR/(len(Test0)+len(Test1)))

