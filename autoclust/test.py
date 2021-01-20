from autoclust import auto_clust
from autoclust import P
from scipy.io import arff
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

data = arff.loadarff('2.arff')

df = pd.DataFrame(data[0])

x = df.x
y = df.y
label = df.label

colors = cm.rainbow(np.linspace(0, 1, len(label)))


fig, axs = plt.subplots(4)
axs[0].scatter(x, y, c=colors)

points = []

for i, row in df.iterrows():
    points.append(P(row.x, row.y))

result1, result2, result3 = auto_clust(points)

x = []
y = []
label = []

for curr in result1:
    x.append(curr.x)
    y.append(curr.y)
    label.append(curr.label)
colors = cm.rainbow(np.linspace(0, 1, len(set(label))))
axs[1].scatter(x, y, c=label, label=set(label))
axs[1].legend()
print("First!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(set(label))
for i in range(0,len(set(label))):
    print("Dla labela " + str(i) + " mamy " + str(len([x for x in result1 if x.label is i] )) + "vertexów")


x = []
y = []
label = []

for curr in result2:
    x.append(curr.x)
    y.append(curr.y)
    label.append(curr.label)
colors = cm.rainbow(np.linspace(0, 1, len(set(label))))
axs[2].scatter(x, y, c=label)
print("second!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(set(label))
for i in range(0,len(set(label))):
    print("Dla labela " + str(i) + " mamy " + str(len([x for x in result2 if x.label is i] )) + "vertexów")

x = []
y = []
label = []
for curr in result3:
    x.append(curr.x)
    y.append(curr.y)
    label.append(curr.label)
colors = cm.rainbow(np.linspace(0, 1, len(label)))
axs[3].scatter(x, y, c=colors)
print("third!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(set(label))
for i in range(0,len(set(label))):
    print("Dla labela " + str(i) + " mamy " + str(len([x for x in result3 if x.label is label[i] ])) + "vertexów")
for i in range(1, len(x)):
    axs[3].text(x[i],y[i],str(label[i]))
plt.show()