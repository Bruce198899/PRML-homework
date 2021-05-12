import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
data = pd.read_excel('city_dist.xlsx', engine='openpyxl').values
dist_mat = data[:, 1:]
city_name = data[:, 0]
mds = MDS(dissimilarity='precomputed')
location = mds.fit_transform(dist_mat)
plt.figure(figsize=(8, 8))
plt.axis('equal')
plt.scatter(location[:, 0], location[:, 1])
for i in range(len(city_name)):
    plt.text(location[i, 0]+0.5, location[i, 1], city_name[i])
plt.show()


