import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'


def shiftMatr(vec):
    mtr = np.array([[1, 0, vec[0]], [0, 1, vec[1]], [0, 0, 1]])
    return mtr


def rotMatr(ang):
    mtr = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return mtr


def scalMatr(a):
    mtr = np.array([[a, 0, 0], [0, a, 0], [0, 0, 1]])
    return mtr


def transfMatr(x, y):
    mtr = np.array([[0, 0, y], [0, 0, x], [0, 0, 0]])
    # mtr = np.array([x, y])
    return mtr


def to_proj_coords(x):
    r, c = x.shape
    x = np.concatenate([x, np.ones((1, c))], axis=0)
    return x


def to_cart_coords(x):
    x = x[:-1] / x[-1]
    return x


def bresenham(x0, y0, x1, y1, n):
    steps_num = int(np.max([np.abs(x0 - x1), np.abs(y0 - y1)]))
    sp = np.linspace(0, 1, steps_num + 1)

    x_coords = np.int32(np.round(x0 * sp + x1 * (1 - sp)))
    y_coords = np.int32(np.round(y0 * sp + y1 * (1 - sp)))

    x_ind = (x_coords > 0) & (x_coords < n)
    y_ind = (y_coords > 0) & (y_coords < n)
    ind = x_ind & y_ind

    x_coords = x_coords[ind]
    y_coords = y_coords[ind]
    res = [list(a) for a in zip(x_coords, y_coords)]
    return (res)


def find_third_point(x1, y1, x2, y2):
    return [(x1 + x2 + - (y1 - y2) * np.sqrt(3)) / 2, (y1 + y2 + - (x2 - x1) * np.sqrt(3)) / 2]


def line_len(x0, y0, x1, y1):
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def main():
    N = 100  # кол-во кадров
    size = 300  # размер картинки
    frames = []
    fig = plt.figure()

    color0 = np.array([255, 102, 178], dtype=np.uint8)
    color1 = np.array([178, 102, 255], dtype=np.uint8)
    color2 = np.array([102, 255, 178], dtype=np.uint8)

    # задаём начальные данный квадрата
    width = np.arange(start=size // 10, stop=9 * size // 10)
    height = np.arange(start=size // 10, stop=9 * size // 10)

    # задаём начальные данные окружности
    theta = np.linspace(0, size * np.pi, size)
    r = size // 5
    pos = size // 2
    x_circle = list(map(int, r * np.cos(theta) + pos))
    y_circle = list(map(int, r * np.sin(theta) + pos))

    # задаём начальные данные для треугольника
    accessible_pts = list()
    for i in range(size):
        for j in range(size):
            if (i < width[-1] and i > width[0] and
                j < height[-1] and j > height[0] and
                line_len(i, j, pos, pos) > r):
                accessible_pts.append([i, j])

    # случайно задаём треуголник фиксированного размера на поле между квадратом и окружностью
    pt = [random.choice(accessible_pts)]
    pt.append([pt[0][0]+20, pt[0][1]])
    if pt[1] not in accessible_pts:
        pt[1] = [pt[0][0]-20, pt[0][1]]
    pt.append(find_third_point(pt[0][0], pt[0][1], pt[1][0], pt[1][1]))
    if pt[2] not in accessible_pts:
        pt[2] = [pt[2][0], 2*pt[0][1]-pt[2][1]]
    center = np.array([(pt[0][0] + pt[1][0] + pt[2][0]) / 3, (pt[0][1] + pt[1][1] + pt[2][1]) / 3])
    m = len(pt)

    x = np.array(pt, dtype=np.float32).T
    x_proj = to_proj_coords(x)

    theta0 = random.choice(np.linspace(0, 2 * np.pi))
    step_x = 3 * np.cos(theta0)
    step_y = 3 * np.sin(theta0)
    trans_x = -1 * step_x
    trans_y = -1 * step_y
    count0 = 0

    # заполнение массива кадров
    i = 0
    while i < (2 * N):
        change = 0
        count1 = 0
        # создаем кадр
        img = np.full((size, size, 3), 255, dtype=np.uint8)

        # рисуем квадрат
        img[width, height[0]] = color0
        img[width, height[len(height) - 1]] = color0
        img[width[0], height] = color0
        img[width[len(width) - 1], height] = color0

        # рисуем окружность
        img[x_circle, y_circle] = color1

        # получаем координаты новых положений граней треугольника
        T = shiftMatr(-center)
        R = rotMatr(i % (N // 2) * 2 * np.pi / (N // 2))
        S = (scalMatr(1 + i % N / (N // 2)) if i % N <= (N // 2) else scalMatr(3 - i % N / (N // 2)))
        trans_x += step_x
        trans_y += step_y
        B = transfMatr(trans_x, trans_y)
        x_new = to_cart_coords(np.linalg.inv(T) @ (R @ S + B) @ T @ x_proj)

        # рисуем треугольник
        for j in range(m):
            line_points = bresenham(x_new[0, j], x_new[1, j], x_new[0, (j+1)%m], x_new[1, (j+1)%m], size)
            for z in range(len(line_points)):
                if np.all(img[line_points[z][0], line_points[z][1]] == color0) or np.all(
                        img[line_points[z][0], line_points[z][1]] == color1):
                    if np.all(img[line_points[z][0], line_points[z][1]] == color0):
                        change = 1
                    else:
                        change = 2
                img[line_points[z][0], line_points[z][1]] = color2
                if line_points[z][0] >= width[-1] or line_points[z][0] <= width[0]:
                    count1 += 1
                elif line_points[z][1] >= height[-1] or line_points[z][1] <= height[0]:
                    count1 += 1
                elif line_len(line_points[z][0], line_points[z][1], pos, pos) <= r:
                    count1 += 1

        # меняем цвета и напрвление движения треугольника, если он столкнулся с другими фигурами
        if change:
            if count0 == 0:
                if change == 1:
                    color0, color2 = color2, color0
                else:
                    color1, color2 = color2, color1
                theta0 = random.choice(np.linspace(0, 2 * np.pi))
                step_x = 3 * np.cos(theta0)
                step_y = 3 * np.sin(theta0)
                trans_x0 = trans_x
                trans_y0 = trans_y
                im = plt.imshow(img)
                frames.append([im])
            if count0 > 0:
                if count1 >= count0:
                    trans_x = trans_x0
                    trans_y = trans_y0
                    theta0 = random.choice(np.linspace(0, 2 * np.pi))
                    step_x = 3 * np.cos(theta0)
                    step_y = 3 * np.sin(theta0)
            count0 = count1

        if not change:
            count0 = 0
            i += 1
            im = plt.imshow(img)
            frames.append([im])

    print('Frames creation finshed.')

    # mp4 animation creation
    ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('animation.mp4', writer)

    # gif animation creation
    ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
    writer = PillowWriter(fps=24)
    ani.save('animation.gif', writer=writer)

    plt.show()


if __name__ == '__main__':
    main()
