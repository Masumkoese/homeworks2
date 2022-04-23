import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



def f(x,y):
    i = 1j
    return (math.e**(i*y))+((2*(math.e)**((-i)*y/2))*math.cos(math.sqrt(3)*x/2))

eigv1 = []
eigv2 = []
xp = []
yp = []


z = np.arange(-math.pi,math.pi,.1)
for x in z:
    for y in z:

        xp.append(x)
        yp.append(y)

        M = np.array([[0, f(x, y)], [f(x, y).conjugate(), 0]], dtype='complex_')
        print(M)

        eigenvalues, eigenvectors = np.linalg.eig(M)
        P = eigenvectors

        D = np.zeros((2, 2), dtype='complex_')
        for m in range(2):
            D[m, m] = eigenvalues[m]

        P_inv = np.linalg.inv(P)

        print(P.dot(D.dot(P_inv)))

        eigv1.append(eigenvalues[0])
        eigv2.append(eigenvalues[1])



fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')

ax.scatter(xp, yp, eigv1,label = 'Eigenvalue 1', c = 'r', s = 50)
ax.scatter(xp, yp, eigv2,label = 'Eigenvalue 2', c = 'b', s = 50)

ax.set_title('3D Scatter Plot')


ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
plt.savefig('3d_scatter.png')
plt.show()





