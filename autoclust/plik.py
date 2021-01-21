from autoclust import auto_clust
from autoclust import P
from scipy.io import arff
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

class LabledPoint:
  def __init__(self, x, y, label):
    self.x = x
    self.y = y
    self.label = label


data = arff.loadarff('a.arff')

df = pd.DataFrame(data[0])

points = []

for i, row in df.iterrows():
    points.append(P(row.x, row.y))

result1, result2, result3 = auto_clust(points)

points = []

for curr in result3:
    points.append(LabledPoint(curr.x, curr.y, curr.label))


points.sort(key=lambda l: l.label)

oldLabel = points[0].label
newLabel = 0


for point in points:
    if oldLabel is point.label:
        point.label = newLabel
    else:
        oldLabel = point.label
        newLabel += 1
        point.label = newLabel


label = []
x = []
y = []
for i in range(1,len(points)):
    x.append(points[i].x)
    y.append(points[i].y)
    label.append(points[i].label)
fig=plt.figure()
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0,len(set(label)),len(set(label)) + 1)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# make the scatter
plt.scatter(x,y,c=label,cmap=cmap, norm=norm)
for i in range(1, len(x)):
    if i%1 is 0:
        plt.text(x[i],y[i],str(label[i]))

plt.show()
