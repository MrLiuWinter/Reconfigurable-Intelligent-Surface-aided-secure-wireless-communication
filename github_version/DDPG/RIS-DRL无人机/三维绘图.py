import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax = fig.gca(projection='3d')
BS=np.array([
        [0,0,50]
])
IoT=np.array([
        [800,800],
        [900,900],
        [1000,1000]
])
ax.scatter(BS[0,0], BS[0,1],BS[0,2] ,c='r', marker='x',s = 40,label="RIS")
ax.scatter(IoT[:,0], IoT[:,1], c='c', marker='x',s = 40,label="GT")

UAV=np.array([
        [0,0,150],
        [100,100,150],
        [200,200,150],
        [400,300,150],
[400,300,200],

])
ax.plot(UAV[:,0], UAV[:,1], UAV[:,2], c='g',linestyle='-', marker='', label=u"RIS-Assisted UAV")
ax.set_zlim(0, 250)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend( loc='upper right', shadow=True)
plt.show()
