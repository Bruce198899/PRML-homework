import numpy as np
import matplotlib.pyplot as plt
_Sample = 1000
_Mondai = 3  # 该参数对应8.3中第几个问题
if _Mondai == 1:
    for _iter in range(3):
        X = np.random.normal(0, 1, _Sample)
        Miu = np.mean(X)
        Sigma2 = np.sum((X-Miu)*(X-Miu))/_Sample
        Sigma = np.sqrt(Sigma2)
        x = np.linspace(Miu-4*Sigma, Miu+4*Sigma, 100)
        y = np.exp(-(x-Miu)**2/(2*Sigma2)) / (np.sqrt(2*np.pi)*Sigma)
        plt.plot(x, y)
    x = np.linspace(-5, 5, 100)
    y = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(x, y, "r", linewidth=1)
    plt.title('Red:Standard Normal Distribution (Sample = {})'.format(_Sample))
    plt.show()

elif _Mondai == 2:
    X = np.random.normal(0, 1, 1000)
    SigmaRatio = [0.01, 0.1, 1, 10]
    for i in range(len(SigmaRatio)):
        Miu = (np.sum(X)-5/SigmaRatio[i])/(1000+1/SigmaRatio[i])
        x = np.linspace(Miu - 5, Miu + 5, 100)  # 定义域
        y = np.exp(-(x-Miu)**2/2)/(np.sqrt(2*np.pi))  # 定义曲线函数
        plt.plot(x, y)  # 加载曲线
    x = np.linspace(-5, 5, 100)
    y = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(x, y, "r", linewidth=1)
    plt.title('Red:Standard Normal Distribution (Sample = {})'.format(1000))
    plt.show()
else:
    for _iter in range(3):
        X = np.random.uniform(0, 1, 100)
        Miu = np.mean(X)
        Sigma2 = np.sum((X-Miu)*(X-Miu))/100
        Sigma = np.sqrt(Sigma2)
        x = np.linspace(Miu-4*Sigma, Miu+4*Sigma, 100)
        y = np.exp(-(x-Miu)**2/(2*Sigma2)) / (np.sqrt(2*np.pi)*Sigma)
        plt.plot(x, y)
    plt.plot([-1, -0.00001, 0, 1, 1.00001, 2], [0, 0, 1, 1, 0, 0], "r", linewidth=1)
    plt.title('Red:Uniform Distribution (Sample = {})'.format(100))
    plt.show()
