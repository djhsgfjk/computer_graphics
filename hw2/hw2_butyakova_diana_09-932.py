import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave


def list_v_fs(v_amount, f_amount, faces):
    v_fs = [[] for i in range(v_amount)]
    for i in range(f_amount):
        for v in faces[i]:
            v_fs[v].append(i)
    return (v_fs)


def list_edges(f_amount, v_fs, faces):
    edges = list()
    for f0 in range(f_amount):
        for i in range(len(faces[f0])):
            for j in range(i + 1, len(faces[f0])):
                v1 = faces[f0][i]
                v2 = faces[f0][j]
                if v1 > v2:
                    v1, v2 = v2, v1
                if not (v1, v2) in edges:
                    if set(v_fs[v1]) & set(v_fs[v2]):
                        edges.append((v1, v2))
    return (edges)


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


def line_len(x0, y0, x1, y1):
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def find_color(x, y, n, base_color):
    d = line_len(n // 2, n // 2, x, y)
    res = base_color * (1 - d / n)
    return (res)


def set_color(image, x, y, color):
    image[x, y, :] = color
    return (image)


def draw_line(image, x0, y0, x1, y1, n, base_color):
    line = bresenham(x0, y0, x1, y1, n)
    for x, y in line:
        color = find_color(x, y, n, base_color)
        image = set_color(image, x, y, color)
    return (image)


def create_image(heigh, widht, background_color):
    image = np.zeros((heigh, widht, 3), np.uint8)
    image[:, :, :] = background_color
    return (image)


def show_image(image):
    image = np.flipud(image)  # потому что ось OY в обратном направлении
    plt.figure()
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    image = np.flipud(image)
    imsave(name, image)


def main():
    # читаем инф-ю из файла и записываем списки
    vertices = list()
    faces = list()
    f = open('teapot.obj', 'r')
    line = f.readline()
    while line[0] == 'v':
        vertices.append(list(map(float, line[2:].split())))
        line = f.readline()
    while not line[0] == 'f':
        line = f.readline()
    while not line == '':
        faces.append(list(i - 1 for i in list(map(int, line[2:].split()))))
        line = f.readline()
    f.close()

    v_amount = len(vertices)
    f_amount = len(faces)

    # создаем список "номер вершины:номера граней, к которым принадлежит эта вершина"
    v_fs = list_v_fs(v_amount, f_amount, faces)
    # создаем список ребер
    edges = list_edges(f_amount, v_fs, faces)

    # начинаем работу с рисунком
    # находим макчимальную ширину(длину) рисунка и округляем её
    m = math.ceil(max(max(vertices[i][0] for i in range(v_amount))
                      - min(vertices[i][0] for i in range(v_amount)),
                      max(vertices[i][1] for i in range(v_amount))
                      - min(vertices[i][1] for i in range(v_amount))))
    n = 1000
    background_color = np.array((255, 255, 255), dtype=np.uint8)
    base_color = np.array((255, 20, 147), dtype=np.uint8)
    image = create_image(n, n, background_color)

    # рисуем картинку
    for e in edges:
        # находим координаты начала и конца грани
        # масштабируем
        # параллельно переносим в центр изображения
        # поворачиваем на 90 градусов и отражаем (матрица поворота * отражения равна (0 1, 1 0))
        x0 = vertices[e[0]][1] * n / m + n / 2
        y0 = vertices[e[0]][0] * n / m + n / 2
        x1 = vertices[e[1]][1] * n / m + n / 2
        y1 = vertices[e[1]][0] * n / m + n / 2
        image = draw_line(image, x0, y0, x1, y1, n, base_color)

    show_image(image)
    save_image(image, 'teapot.jpg')


if __name__ == '__main__':
    main()
