import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'

N = 26
R = 40
size = 1.5 * R

fig = plt.figure(figsize=(5, 5))
ax = plt.axes(xlim=(-size, size), ylim=(-size, size))
curve, = ax.plot([], [], lw=3)
curve.set_color('lightpink')


def init():
    curve.set_data([], [])
    return curve,


def animate(i):
    t = abs(((i % 21) - 10) / 10) + 0.5
    x, y = creat_curve(N, R, t)
    curve.set_data(x, y)
    return curve,


def algorithm_De_Casteljau(x0, y0, x1, y1, x2, y2):
    x, y = [], []
    arr = np.arange(0, 1, 0.05)
    for t in arr:
        x01 = x0 + (x1 - x0) * t
        y01 = y0 + (y1 - y0) * t
        x12 = x1 + (x2 - x1) * t
        y12 = y1 + (y2 - y1) * t
        x.append(x01 + (x12 - x01) * t)
        y.append(y01 + (y12 - y01) * t)

    return x, y


def creat_curve(N, R, t):
    x_control = np.zeros((N, 3), dtype=float)
    y_control = np.zeros((N, 3), dtype=float)
    for i in range(N):
        x1 = R * (t + (i % 2) * 2 * (1 - t)) * np.cos(i * (2 * np.pi / N))
        y1 = R * (t + (i % 2) * 2 * (1 - t)) * np.sin(i * (2 * np.pi / N))
        x_control[i][1] = x1
        y_control[i][1] = y1
    for i in range(N):
        x_control[i][0] = (x_control[(N + i - 1) % N, 1] + x_control[i, 1]) / 2
        x_control[(N + i - 1) % N, 2] = x_control[i, 0]
        y_control[i][0] = (y_control[(N + i - 1) % N, 1] + y_control[i, 1]) / 2
        y_control[(N + i - 1) % N, 2] = y_control[i, 0]
    x, y = [], []
    for i in range(N):
        x_new, y_new = algorithm_De_Casteljau(x_control[i, 0], y_control[i, 0],
                                              x_control[i, 1], y_control[i, 1],
                                              x_control[i, 2], y_control[i, 2])
        x += x_new
        y += y_new
    x.append(x_control[0, 0])
    y.append(y_control[0, 0])
    return x, y


def main():
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100)
    writer = animation.PillowWriter(fps=80)
    ani.save("bezier_curve.gif", writer=writer)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("bezier_curve.mp4", writer=writer)
    plt.show()


if __name__ == '__main__':
    main()
