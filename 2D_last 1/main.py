import numpy as np
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from skimage import metrics
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

'''def j2D(x_sup, x_1):
    j_max = 0
    max = 0
    for j in range(len(x_sup)):
        loc = Hausdorff_d(np.array([x_sup[j]]), x_1)
        a = np.append(x_1, [x_sup[j]], axis=0)
        if (loc >= max):
            j_max = j
            max = loc
    return np.append(x_1, [x_sup[j_max]], axis=0), np.delete(x_sup, j_max, axis=0)'''


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


def print_polygon(y, y_2):
    plt.figure()
    if (len(y_2)>2):
        polx = Polygon(y_2).convex_hull
        x1, y1 = polx.exterior.xy
        plt.plot(x1, y1, color='#de3163', alpha=0.7,
                 linewidth=5, solid_capstyle='round', zorder=2)
    elif (len(y_2)==2):
        polx = LineString(y_2)
        plt.plot(*polx.xy, color='#de3163', alpha=0.7,
                 linewidth=5, solid_capstyle='round', zorder=2)
    else:
        polx = Point(y_2)
        plt.plot(*polx)

    poly = Polygon(y)

    x, y = poly.exterior.xy

    plt.plot(x, y, color='#423189', alpha=0.7,
             linewidth=5, solid_capstyle='round', zorder=2)



    plt.grid(True)


    print(np.sqrt(np.abs(np.max(fin))), res)




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

eps = 0.01
import pandas as pd
import openpyxl
import warnings

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    df = pd.read_excel("1MN-120.xlsx", header=None)

y = np.array(df.to_numpy(dtype=float))

#y = np.array([[2, 0], [0, 1], [-1, 1], [-2, 0], [-0.9, -1.9], [0, -3], [1, -2]])
#y = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
#x = np.array([[0.9, 0]])
#y = np.array ([[-1, 0], [1, 0], [0, 1], [0, -1], [np.cos(7*np.pi/8), np.sin(7*np.pi/8)], [np.cos(7*np.pi/8), -np.sin(7*np.pi/8)]])
#y = y - x

k = np.array([])
dp = np.array([])
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


s = 1
y_sup = y
y_2 = np.array([y_sup[i_max]])
y_sup = np.delete(y_sup, i_max, axis=0)
d = np.sqrt(np.abs(np.max(Hausdorff_d(y_2, y))))
dp = np.append(dp, d)
k = np.append(k, s)


while (d >= eps and len(y_sup) != 0):
    #print("len:", k)
    y_2, y_sup = j2D(y_sup, y_2, y)
    d = np.sqrt(np.abs(np.max(Hausdorff_d(y_2, y))))
    dp = np.append(dp, d)
    k = np.append(k, len(y_2))
    #print_polygon(y, y_2)

print("len**:", len(y_2))
res = Algo(y_2, y)
#res = y_2

fin = (Hausdorff_d(res, y))



plt.show()

#вывод

poly = Polygon(y).convex_hull
polx = Polygon(res).convex_hull
x, y = poly.exterior.xy
x1, y1 = polx.exterior.xy



plt.plot(x, y, color='#423189', alpha=0.7,
    linewidth=5, solid_capstyle='round', zorder=2)

plt.plot(x1, y1, color='#de3163', alpha=0.7,
    linewidth=5, solid_capstyle='round', zorder=3)

fig_for_graphs = plt.figure(figsize=[13,13])
ax_for_graphs = fig_for_graphs.add_subplot(1,1,1)
ax_for_graphs.plot(k,dp,color='blue')
plt.ylabel("epsilon(I_(k)'')")
plt.xlabel("k")
#ax_for_graphs.set_title("d(k)")
#ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

plt.grid(True)
plt.show()



print(np.sqrt(np.abs(np.max(fin))), res, '\n', len(res))

