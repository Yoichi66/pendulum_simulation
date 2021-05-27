# https://takun-physics.net/12612/
#from numpy import sin, cos
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import pandas as pd


df_param = pd.read_csv('N_param0.csv')
df_param = pd.DataFrame(df_param)
#df_param = df_param[:, 1:].values
df_param = np.array(df_param)
df_param = df_param[:, 1:]

m = np.array(df_param[0, :]).astype(np.float64).round(2)
L = np.array(df_param[1, :]).astype(np.float64).round(2)
theta = np.array(df_param[2, :]).astype(np.float64).round(2)
x = np.array([np.radians(i) for i in theta]).astype(np.float64)
v = np.array(df_param[3, :]).astype(np.float64).round(2)

# 初期条件のコピー
x0 = x.copy()

# 初期位置の確認
n = len(df_param[0, :])

x_ini_cor = np.zeros(n, dtype=np.float64)
y_ini_cor = np.zeros(n, dtype=np.float64)


# 微分方程式
def ini_cor_func(j):
    if j == 0:
        x_ini_cor[j] = L[j] * np.sin(x[j])
        y_ini_cor[j] = -L[j] * np.cos(x[j])
    else:
        x_ini_cor[j] = L[j] * np.sin(x[j]) + x_ini_cor[j - 1]
        y_ini_cor[j] = -L[j] * np.cos(x[j]) + y_ini_cor[j - 1]

    return x_ini_cor[j], y_ini_cor[j]


for j in range(n):
    x_ini_cor[j], y_ini_cor[j] = ini_cor_func(j)

x_ini_cor = x_ini_cor * 1000
y_ini_cor = y_ini_cor * 1000

xplot_ = np.insert(x_ini_cor, 0, 0)
yplot_ = np.insert(y_ini_cor, 0, 0)

plt.grid()
plt.plot(xplot_, yplot_, 'ko-', lw=2)

# Calculate propertv
init = 0
end = 20
dt = 0.05
h = dt
loop = int(end / h)

n = len(df_param[0, :])
g = 9.8

# initial state
t = init

tpoints = np.arange(init, end, h)
xpoints = []
vpoints = []

# A = np.zeros((n,n),dtype=np.float64)
# B = np.zeros((n,n),dtype=np.float64)

E = -np.ones_like(x)


def N_func(t, x, v):
    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            for k in range(max(i, j), n):
                A[i][j] += m[k]
                B[i][j] += m[k]
            if i == j:
                A[i][j] *= L[j]
                B[i][j] *= g * np.sin(x[i])
            else:
                A[i][j] *= L[j] * np.cos(x[i] - x[j])
                B[i][j] *= L[j] * v[j] ** 2 * np.sin(x[i] - x[j])

    # 逆行列の計算
    inv_A = np.linalg.inv(A)

    # inv_A*Bを計算
    inv_A_B = np.dot(inv_A, B)

    F = np.dot(inv_A_B, E)

    return F


xpoints = []
vpoints = []

# 配列要素数の定義
j1 = np.zeros_like(v)
k1 = np.zeros_like(x)

j2 = np.zeros_like(v)
k2 = np.zeros_like(x)

j3 = np.zeros_like(v)
k3 = np.zeros_like(x)

j4 = np.zeros_like(v)
k4 = np.zeros_like(x)


def RK(t, x, v):
    vt = v.copy()
    xt = x.copy()
    xpoints.append(xt)
    vpoints.append(vt)

    j1 = N_func(t, x, v) * h
    k1 = v * h

    j2 = N_func(t + h / 2, x + k1 / 2, v + j1 / 2) * h
    k2 = (v + j1 / 2) * h

    j3 = N_func(t + h / 2, x + k2 / 2, v + j2 / 2) * h
    k3 = (v + j2 / 2) * h

    j4 = N_func(t + h, x + k3, v + j3) * h
    k4 = (v + j3) * h

    v += (j1 + 2 * j2 + 2 * j3 + j4) / 6
    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, v, xpoints, vpoints


# from ipykernel import kernelapp as app

for t in range(len(tpoints)):
    # for t in range(2):
    x, v, xpoints, vpoints = RK(t, x, v)

xpoints = np.array(xpoints)
vpoints = np.array(vpoints)

xt = np.zeros(len(xpoints))


def xt_func(j):
    xt = xpoints[:, j]
    return xt


x_cor = np.zeros((len(xpoints), n), dtype=np.float64)
y_cor = np.zeros((len(xpoints), n), dtype=np.float64)


def cor_func(j):
    if j == 0:
        x_cor[:, j] = L[j] * np.sin(xt_func(j))
        y_cor[:, j] = -L[j] * np.cos(xt_func(j))
    else:
        x_cor[:, j] = L[j] * np.sin(xt_func(j)) + x_cor[:, j - 1]
        y_cor[:, j] = -L[j] * np.cos(xt_func(j)) + y_cor[:, j - 1]

    return x_cor[:, j], y_cor[:, j]


for j in range(n):
    x_cor[:, j], y_cor[:, j] = cor_func(j)

x_cor = x_cor * 1000
y_cor = y_cor * 1000

for j in range(n):
    plt.plot(x_cor[:,j], y_cor[:,j])


# generate animation

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-n * 1000, n * 1000),
                     ylim=(-n * 1000 - n * 1000 * 0.5, n * 1000 - n * 1000 * 0.5))
# xlim=(-n*1000, n*1000), ylim=(-n*1000-n*1000*0.5, n*1000-n*1000*0.5)
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = []
    thisy = []
    thisx.append(0)
    thisy.append(0)
    for j in range(n):
        thisx.append(x_cor[i, j])
        thisy.append(y_cor[i, j])

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, np.arange(1, len(tpoints)),
                              interval=25, blit=True, init_func=init)

# ani.save('N_furiko_spring.mp4', fps=15)
writergif = animation.PillowWriter(fps=30)
ani.save('N_pendulum.gif', writer=writergif)
plt.show()

