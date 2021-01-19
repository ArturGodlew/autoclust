from autoclust import auto_clust
from autoclust import P

from scipy.io import arff
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import importlib
import autoclust
importlib.reload(autoclust)

data = arff.loadarff('data.arff')

df = pd.DataFrame(data[0])

x = df.a0
y = df.a1
label = df.CLASS

fig, axs = plt.subplots(2)
axs[0].scatter(x, y, c=label)

points = []

for i, row in df.iterrows():
    points.append(P(row.a0, row.a1))

result = auto_clust(points)

x = []
y = []
label = []
for curr in result:
    x.append(curr.x)
    y.append(curr.y)
    label.append(curr.label)


axs[1].scatter(x, y, c=label)

plt.show()