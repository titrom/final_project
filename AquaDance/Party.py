import Ball
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def diff(ii,x1,y1):
    xs,ys=[],[]
    for h in range(len(balls), len(y1), len(balls)):
        ys.append(y1[h - len(balls):h])
    for h in range(len(balls), len(x1), len(balls)):
        xs.append(x1[h - len(balls):h])
    for j in range(0, len(balls)):
        if np.abs(balls[j].get_x() - X1) <= radius or np.abs(balls[j].get_x() - X2) <= radius:
            balls[j].set_vx(-balls[j].get_vx())
        if np.abs(balls[j].get_y() - Y1) <= radius or np.abs(balls[j].get_y() - Y2) <= radius:
            balls[j].set_vy(-balls[j].get_vy())
        for d in  range(j,len(balls)):
            if j != d:
                r1 = np.sqrt((xs[ii][j] - xs[ii][d]) ** 2 + (ys[ii][j] - ys[ii][d]) ** 2)
                r0 = np.sqrt((xs[ii - 1][j] - xs[ii - 1][d]) ** 2 + (ys[ii- 1][j] - ys[ii - 1][d]) ** 2)
                if r1 <= radius * 2 < r0:
                    res = collision(balls[j].get_x(), balls[j].get_vx(), balls[j].get_y(), balls[j].get_vy(),
                                    balls[j].get_mass(), balls[d].get_x(), balls[d].get_vx(),
                                    balls[d].get_y(), balls[ d].get_vy(), balls[d].get_mass())
                    balls[j].set_vx(res[0])
                    balls[j].set_vy(res[1])
                    balls[d].set_vx(res[2])
                    balls[d].set_vy(res[3])


def circle_function(x_central, y_central, radius):
    x = np.zeros(80)
    y = np.zeros(80)
    for p in range(0, 80, 1):
        alpha = np.linspace(0, 2 * np.pi, 80)
        x[p] = x_central + (radius - 0.05) * np.cos(alpha[p])
        y[p] = y_central + (radius - 0.05) * np.sin(alpha[p])
    return x, y


def collision(x_1, vx1, y_1, vy1, mass1, x_2, vx2, y_2, vy2, mass2):
    r12 = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
    v2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

    if r12 <= (radius * 2):

        if v1 != 0:
            theta1 = np.arccos(vx1 / v1)
        else:
            theta1 = 0
        if v2 != 0:
            theta2 = np.arccos(vx2 / v2)
        else:
            theta2 = 0
        if vy1 < 0:
            theta1 = - theta1 + 2 * np.pi
        if vy2 < 0:
            theta2 = - theta2 + 2 * np.pi

        if (y_1 - y_2) < 0:
            phi = - np.arccos((x_1 - x_2) / r12) + 2 * np.pi
        else:
            phi = np.arccos((x_1 - x_2) / r12)

        vx_1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
               * np.cos(phi) / (mass1 + mass2) \
               + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi)) \
               * np.cos(phi) / (mass1 + mass2) \
               + K * v1 * np.sin(theta1 - phi) * np.cos(phi + np.pi / 2)

        vy_1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
               * np.sin(phi) / (mass1 + mass2) \
               + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi)) \
               * np.sin(phi) / (mass1 + mass2) \
               + K * v1 * np.sin(theta1 - phi) * np.sin(phi + np.pi / 2)

        vx_2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
               * np.cos(phi) / (mass1 + mass2) \
               + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
               * np.cos(phi) / (mass1 + mass2) \
               + K * v2 * np.sin(theta2 - phi) * np.cos(phi + np.pi / 2)

        vy_2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
               * np.sin(phi) / (mass1 + mass2) \
               + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
               * np.sin(phi) / (mass1 + mass2) \
               + K * v2 * np.sin(theta2 - phi) * np.sin(phi + np.pi / 2)
    else:
        vx_1, vy_1, vx_2, vy_2 = vx1, vy1, vx2, vy2

    return vx_1, vy_1, vx_2, vy_2


def move_func(start, t):
    x_1, v_x1, y_1, v_y1, x_2, v_x2, y_2, v_y2, x_3, v_x3, y_3, v_y3\
        , x_4, v_x4, y_4, v_y4, x_5, v_x5, y_5, v_y5 , x_6, v_x6, y_6, v_y6, x_0, v_x0, y_0, v_y0 = start


    dx1_dt = v_x1
    dvx1_dt = 0

    dy1_dt = v_y1
    dvy1_dt = 0

    dx2_dt = v_x2
    dvx2_dt = 0

    dy2_dt = v_y2
    dvy2_dt = 0

    dx3_dt = v_x3
    dvx3_dt = 0

    dy3_dt = v_y3
    dvy3_dt = 0

    dx4_dt = v_x4
    dvx4_dt = 0

    dy4_dt = v_y4
    dvy4_dt = 0

    dx5_dt = v_x5
    dvx5_dt = 0

    dy5_dt = v_y5
    dvy5_dt = 0

    dx6_dt = v_x6
    dvx6_dt = 0

    dy6_dt = v_y6
    dvy6_dt = 0

    dx0_dt = v_x0
    dvx0_dt = 0

    dy0_dt = v_y0
    dvy0_dt = 0


    return dx1_dt, dvx1_dt, dy1_dt, dvy1_dt, dx2_dt, dvx2_dt, dy2_dt, dvy2_dt, dx3_dt, dvx3_dt, dy3_dt, dvy3_dt\
        , dx4_dt, dvx4_dt, dy4_dt, dvy4_dt, dx5_dt, dvx5_dt, dy5_dt, dvy5_dt, dx6_dt, dvx6_dt, dy6_dt, dvy6_dt\
        , dx0_dt, dvx0_dt, dy0_dt, dvy0_dt



T = 10
N = 2000
K = 1
#Wall
X1 = 1.5
X2 = -1.5
Y1 = -1.5
Y2 = 1.5



radius = 0.4 # Метры
balls = []
ball1 = Ball.Ball(x = -1,vx = 0,y = 1, vy = 0, mass = 1)
ball2 = Ball.Ball(x = 1, vx =0 , y = 1, vy =  0, mass = 1)
ball3 = Ball.Ball(x = -1, vx =  0, y =  0, vy = 0, mass = 1)
ball4 = Ball.Ball(x = -1, vx =  0, y =  -1, vy =  0, mass = 1)
ball5 = Ball.Ball(x = 1, vx =  0, y = -1, vy = 0, mass = 1)
ball6 = Ball.Ball(x = 1, vx = 0, y = 0, vy =  0, mass =  1)
ball0 = Ball.Ball(x = 0, vx = 1, y = 0, vy = 1, mass = 1)



balls.append(ball1)
balls.append(ball2)
balls.append(ball3)
balls.append(ball4)
balls.append(ball5)
balls.append(ball6)

balls.append(ball0)




x, y=[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]

tau = np.linspace(0, T, N)


for i in range(N - 1):
    t = [tau[i], tau[i+1]]
    for j in range(0, len(balls)):

        stat0 = balls[0].get_x(), balls[0].get_vx(), balls[0].get_y(), balls[0].get_vy() \
            , balls[1].get_x(), balls[1].get_vx(), balls[1].get_y(), balls[1].get_vy() \
            , balls[2].get_x(), balls[2].get_vx(), balls[2].get_y(), balls[2].get_vy()\
            , balls[3].get_x(), balls[3].get_vx(), balls[3].get_y(), balls[3].get_vy()\
            , balls[4].get_x(), balls[4].get_vx(), balls[4].get_y(), balls[4].get_vy()\
            , balls[5].get_x(), balls[5].get_vx(), balls[5].get_y(), balls[5].get_vy()\
            , balls[6].get_x(), balls[6].get_vx(), balls[6].get_y(), balls[6].get_vy()


        sol = odeint(move_func, stat0, t)
        balls[0].set_x(sol[1, 0])
        balls[0].set_vx(sol[1, 1])
        balls[0].set_y(sol[1, 2])
        balls[0].set_vy(sol[1, 3])

        balls[1].set_x(sol[1, 4])
        balls[1].set_vx(sol[1, 5])
        balls[1].set_y(sol[1, 6])
        balls[1].set_vy(sol[1, 7])

        balls[2].set_x(sol[1, 8])
        balls[2].set_vx(sol[1, 9])
        balls[2].set_y(sol[1, 10])
        balls[2].set_vy(sol[1, 11])

        balls[3].set_x(sol[1, 12])
        balls[3].set_vx(sol[1, 13])
        balls[3].set_y(sol[1, 14])
        balls[3].set_vy(sol[1, 15])

        balls[4].set_x(sol[1, 16])
        balls[4].set_vx(sol[1, 17])
        balls[4].set_y(sol[1, 18])
        balls[4].set_vy(sol[1, 19])

        balls[5].set_x(sol[1, 20])
        balls[5].set_vx(sol[1, 21])
        balls[5].set_y(sol[1, 22])
        balls[5].set_vy(sol[1, 23])

        balls[6].set_x(sol[1, 24])
        balls[6].set_vx(sol[1, 25])
        balls[6].set_y(sol[1, 26])
        balls[6].set_vy(sol[1, 27])

        x.append(balls[j].get_x())
        y.append(balls[j].get_y())

        diff(i,x,y)







fig, ax = plt.subplots()
ball1, = plt.plot([], [], 'o', color='r', ms=2)
ball2, = plt.plot([], [], 'o', color='r', ms=2)
ball3, = plt.plot([], [], 'o', color='r', ms=2)
ball4, = plt.plot([], [], 'o', color='r', ms=2)
ball5, = plt.plot([], [], 'o', color='r', ms=2)
ball6, = plt.plot([], [], 'o', color='r', ms=2)

ball0, = plt.plot([], [], 'o', color='b', ms=2)




plt.plot([X1,X1],[Y2,Y1], color="b")
plt.plot([X2,X2],[Y2,Y1], color="b")
plt.plot([X2,X1],[Y1,Y1], color="b")
plt.plot([X2,X1],[Y2,Y2], color="b")




def animate(g):
    xs,ys=[],[]
    for h in range(len(balls), len(x), len(balls)):
        xs.append(x[h - len(balls):h])
        ys.append(y[h - len(balls):h])
    ball1.set_data(circle_function(xs[g][0], ys[g][0], radius))
    ball2.set_data(circle_function(xs[g][1], ys[g][1], radius))
    ball3.set_data(circle_function(xs[g][2], ys[g][2], radius))
    ball4.set_data(circle_function(xs[g][3], ys[g][3], radius))
    ball5.set_data(circle_function(xs[g][4], ys[g][4], radius))
    ball6.set_data(circle_function(xs[g][5], ys[g][5], radius))
    ball0.set_data(circle_function(xs[g][6], ys[g][6], radius))


ani = FuncAnimation(fig, animate, frames=N, interval=30)
plt.axis('equal')
plt.ylim(-2, 2)
plt.xlim(-2, 2)
# plt.show()
ani.save("FINAL_PARTY_2.gif", writer="imagemagick", fps = 24)