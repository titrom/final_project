import Ball
import numpy as np
def collision(x_1, vx1, y_1, vy1, mass1, x_2, vx2, y_2, vy2, mass2, K = 1):
    r12 = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
    v2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

    if r12 <= (0.5 * 2):

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
balls = []
ball1 = Ball.Ball(0, 0, 6, -8, 2)
ball2 = Ball.Ball(6, -8, 0, 0, 2)
ball3 = Ball.Ball(-10, -8, 0, 0, 2)

balls.append(ball2)
balls.append(ball3)

T = 10
N = 1000
tau = np.linspace(0, T, N)
mas = []
x = [balls[0].get_x(),balls[1].get_x()]
for i in range(N - 1):
    t = [tau[i], tau[i+1]]
    for j in range(0, len(balls)):
        x.append(balls[j].get_x())
        balls[j].set_x(balls[j].get_x()+1)
print(x)
for h in range(len(balls),len(x),len(balls)):
    mas.append(x[h-len(balls):h])

for i in range(N-1):
    for j in range(0, len(balls)):
        for d in range(j, len(balls)):
            if j != d:
                # print(mas[i][j], mas[i][d])
                print(balls[j].get_vx(),balls[d].get_vx())
                r1 = np.sqrt((mas[i][j] - mas[i][d]) ** 2)
                r0 = np.sqrt((mas[i - 1][j] - mas[i - 1][d]) ** 2)
                if r1 <= 0.5 * 2 < r0:
                    res = collision(balls[j].get_x(), balls[j].get_vx(), balls[j].get_y(), balls[j].get_vy(),
                                    balls[j].get_mass(), balls[d].get_x(), balls[d].get_vx(),
                                    balls[d].get_y(), balls[d].get_vy(), balls[d].get_mass())
                    balls[j].set_vx(res[0])
                    balls[j].set_vy(res[1])
                    balls[d].set_vx(res[2])
                    balls[d].set_vy(res[3])