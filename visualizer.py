import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
fig1 = plt.figure(2)
ax1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure(3)
ax2 = fig2.add_subplot(111, projection='3d')

x = []
y = []
z = []
with open('testdata.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[1]))
        y.append(float(row[2]))
        z.append(float(row[0]))

ax.scatter(x, y, z)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Poynting Coefficient')

a = []
with open('firststep.csv','r') as csvfile2:
    plots = csv.reader(csvfile2, delimiter=',')
    for row in plots:
        a.append(float(row[0]))

ax1.scatter(x, y, a)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Poynting Coefficient')

b = []
with open('tandemres.csv','r') as csvfile2:
    plots = csv.reader(csvfile2, delimiter=',')
    for row in plots:
        b.append(float(row[0]))

ax2.scatter(x, y, b)
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Poynting Coefficient')

plt.show()
