import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
import openpyxl
import warnings
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def Hausdorff_d(x, y):
    res = np.zeros(len(y))
    for j in range(len(y)):
        Z = x.T
        loc = y[j]
        P = matrix(np.dot(Z.T, Z).astype(float))
        P = .5 * (P + P.T)
        q = matrix(-2 * np.dot(Z.T, loc).astype(float))
        # print(q)
        # G = np.array([[1., 2., 1., 1., 4., 6.], [2., 0., 1., 6., 2., 5.], [1., 1., 1., -1., 2., -1.]])
        # h = np.array([3., 2., -2.]).reshape((3,))
        A = matrix(np.ones(len(x))).T
        b = matrix(1.0)
        # G = np.zeros_like(M).astype(float)
        # h = np.zeros_like(M.T[j]).reshape((3,))
        # h = np.array([0., 0., 0.]).reshape((3,))
        G = matrix(-np.eye(len(x)))
        h = matrix(np.zeros(len(x)).reshape((len(x),)))
        solvers.options['show_progress'] = False
        sol = solvers.qp(2 * P, q, G, h, A, b)
        norm_loc = 0
        for k in range(len(loc)):
            norm_loc += loc[k] ** (2)
        res[j] = sol['primal objective'] + norm_loc
    return res

#x_sup - многогранник, в котором убираются вершины
#x_1 - многогранник, в который убираются вершины
def return_ind(x_sup, m):
    i = -1
    for j in range(len(x_sup)):
        if (np.array_equal(m, x_sup[j])):
            i = j
    return i


def j2D(x_sup, x_1, x):
    a = Hausdorff_d(x_1, x)
    max = 0
    max_i = 0
    for i in range(len(a)):
        if (a[i]>=max and x[i] in x_sup):
            m = x[i]
            max = a[i]
            max_i = return_ind(x_sup, m)
    return np.append(x_1, [m], axis=0), np.delete(x_sup, max_i, axis=0)


#y = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], \
     [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, -1, 1], [3/2, 0, 0],\
     [0, 0, 3/2], [0, 3/2, 0], [-3/2, 0, 0], [0, 0, 3/2], [0, -3/2, 0]])
#x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    df = pd.read_excel("Ras.xlsx", header=None).T

y=np.array(df.to_numpy(dtype=float))

#y = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
eps =1.6

def Algo(y_sup1, y_nach):

    min = 1000000
    y_sup = y_sup1
    y = y_nach

    for i in range(len(y)):
        # перебирается каждая вершина изначального массива
        for k in range(len(y_sup)):
            y_1 = np.delete(y_sup, k, axis=0)  # откуда удалять, число, откуда (по строкам)
            # удаляются все вершины по очереди, ищется минимально расстояние
            d = np.sqrt(np.abs(np.max(Hausdorff_d(y_1, y))))
            if (d < min and d <= eps):
                j_min = k
                min = d
        if (min != 1000000):
            print(min, y_sup[j_min])
            y_sup = np.delete(y_sup, j_min, axis=0)
            min = 1000000
        else:
            break
    return y_sup

fin = 0
res = y
loc, i_max, max = 0, 0, 0
for i in range(len(y)):
    for j in range(len(y[i])):
        loc += y[i][j]**2
    if (loc > max):
        max = loc
        i_max = i
    loc = 0

#eps = np.sqrt(max)*0.05


y_sup = y
y_2 = np.array([y_sup[i_max]])
y_sup = np.delete(y_sup, i_max, axis=0)
d = np.sqrt(np.abs(np.max(Hausdorff_d(y_2, y))))
while (d >= eps and len(y_sup) != 0):
    print("len:", len(y_2))
    y_2, y_sup = j2D(y_sup, y_2, y)
    d = np.sqrt(np.abs(np.max(Hausdorff_d(y_2, y))))


res = Algo(y_2, y)


res=y_2
#res = y_2

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
hull = ConvexHull(res)
    # draw the polygons of the convex hull
for s in hull.simplices:
    tri = Poly3DCollection([res[s]])
    tri.set_color('#800080')
    tri.set_alpha(0.5)
    ax.add_collection3d(tri)
    # draw the vertices
ax.scatter(res[:, 0], res[:, 1], res[:, 2], marker='o', color='#32144f')





plt.show()
