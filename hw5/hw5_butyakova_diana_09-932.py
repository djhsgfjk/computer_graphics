import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.image import imsave


def rotMatr(alpha_x, alpha_y, alpha_z):
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(alpha_x), -np.sin(alpha_x), 0],
                   [0, np.sin(alpha_x), np.cos(alpha_x), 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[np.cos(alpha_y), 0, np.sin(alpha_y), 0],
                   [0, 1, 0, 0],
                   [-np.sin(alpha_y), 0, np.cos(alpha_y), 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[np.cos(alpha_z), -np.sin(alpha_z), 0, 0],
                   [np.sin(alpha_z), np.cos(alpha_z), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    R = Rx @ Ry @ Rz
    return R


def shiftMatr(vec):
    T = np.array([[1, 0, 0, vec[0]],
                  [0, 1, 0, vec[1]],
                  [0, 0, 1, vec[2]],
                  [0, 0, 0, 1]])
    return T


def scalMatr(a, b, c):
    S = np.array([[a, 0, 0, 0],
                  [0, b, 0, 0],
                  [0, 0, c, 0],
                  [0, 0, 0, 1]])
    return S


def o2wMatr(alpha_x, alpha_y, alpha_z, vec, a, b, c):
    R = rotMatr(alpha_x, alpha_y, alpha_z)
    T = shiftMatr(vec)
    S = scalMatr(a, b, c)
    Mo2w = T @ R @ S
    return Mo2w


def vec_len(x0, y0, z0, x1, y1, z1):
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)


def RcMatr(x0, y0, z0, x1, y1, z1):
    gamma = np.array([x0 - x1, y0 - y1, z0 - z1]).T / vec_len(x0, y0, z0, x1, y1, z1)
    beta = np.array([0, 1, 0]) - gamma[1] * gamma
    alpha = np.cross(beta, gamma)
    Rc = np.array([[alpha.T[0], beta.T[0], gamma.T[0], 0],
                   [alpha.T[1], beta.T[1], gamma.T[1], 0],
                   [alpha.T[2], beta.T[2], gamma.T[2], 0],
                   [0, 0, 0, 1]])
    return Rc


def w2cMatr(x0, y0, z0, x1, y1, z1):
    Rc = RcMatr(x0, y0, z0, x1, y1, z1)
    Tc = shiftMatr(np.array([-x0, -y0, -z0]))
    Mw2c = Tc @ Rc
    return Mw2c


def orthProjMatr(l, r, b, t, n, f):
    Mproj = np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)],
                      [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                      [0, 0, -2 / (f - n), -(f + n) / (f - n)],
                      [0, 0, 0, 1]])
    return Mproj


def persProjMatr(l, r, b, t, n, f):
    Mproj = np.array([[2 * n / (r - l), 0, (r + l) / (r - l), 0],
                      [0, 2 * n / (t - b), (t + b) / (t - b), 0],
                      [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                      [0, 0, -1, 0]])
    return Mproj


def viewportMatr(x, y, width, height):
    ox = x + width / 2
    oy = y + height / 2
    Tw = shiftMatr(np.array([ox, oy, 1]))
    Sw = scalMatr(width / 2, height / 2, 1)
    Mviewport = Tw @ Sw
    return Mviewport


def to_proj_coords(x):
    r, c = x.shape
    x = np.concatenate([x, np.ones((1, c))], axis=0)
    return x


def to_cart_coords(x):
    x = x[:-1] / x[-1]
    return x


def bresenham(x0, y0, x1, y1, n, m):
    steps_num = int(np.max([np.abs(x0 - x1), np.abs(y0 - y1)]))
    sp = np.linspace(0, 1, steps_num + 1)

    x_coords = np.int32(np.round(x0 * sp + x1 * (1 - sp)))
    y_coords = np.int32(np.round(y0 * sp + y1 * (1 - sp)))

    x_ind = (x_coords > 0) & (x_coords < n)
    y_ind = (y_coords > 0) & (y_coords < m)
    ind = x_ind & y_ind

    x_coords = x_coords[ind]
    y_coords = y_coords[ind]
    res = [list(a) for a in zip(x_coords, y_coords)]
    return (res)


def draw_line(image, x0, y0, x1, y1, n, m, color):
    line = bresenham(x0, y0, x1, y1, n, m)
    for x, y in line:
        image[x, y, :] = color
    return (image)


def barycentric_coordinates(p, p1, p2, p3):
    a = ((p2[1] - p3[1]) * (p[0] - p3[0]) + (p3[0] - p2[0]) * (p[1] - p3[1])) / \
        ((p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1]))
    b = ((p3[1] - p1[1]) * (p[0] - p3[0]) + (p1[0] - p3[0]) * (p[1] - p3[1])) / \
        ((p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1]))
    c = 1.0 - a - b
    return a, b, c


def draw_face(image, v0, v1, v2, color):
    xmin = int(np.floor(min(v0[0], v1[0], v2[0])))
    xmax = int(np.ceil(max(v0[0], v1[0], v2[0])))
    ymin = int(np.floor(min(v0[1], v1[1], v2[1])))
    ymax = int(np.ceil(max(v0[1], v1[1], v2[1])))
    # проверяем принадлежность каждой точки области прямоугольника к грани
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            p = np.array([x, y, 0])
            # проверяем, лежит ли точка внутри треугольников с помощью барицентрических координат
            a, b, c = barycentric_coordinates(p, v0, v1, v2)
            if a >= 0 and b >= 0 and c >= 0:
                z = get_z_coord(p, v0, v1, v2)
                # проверка z-buffer
                if check_z_buffer(x, y, z):
                    image[x, y, :] = color
    return image


def get_z_coord(p, p0, p1, p2):
    x, y, z = p
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    z = (-(x - x0) * (y1 - y0) * (z2 - z0)
         - (y - y0) * (z1 - z0) * (x2 - x0)
         + (x - x0) * (z1 - z0) * (y2 - y0)
         + (y - y0) * (x1 - x0) * (z2 - z0)) / \
        ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) \
        + z0
    return z


def draw_face_texture(image, v0, v1, v2, vt0, vt1, vt2, texture, w, h, base):
    xmin = int(np.floor(min(v0[0], v1[0], v2[0])))
    xmax = int(np.ceil(max(v0[0], v1[0], v2[0])))
    ymin = int(np.floor(min(v0[1], v1[1], v2[1])))
    ymax = int(np.ceil(max(v0[1], v1[1], v2[1])))
    # проверяем принадлежность каждой точки области прямоугольника к грани
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            p = np.array([x, y, 0])
            # проверяем, лежит ли точка внутри треугольников с помощью барицентрических координат
            a, b, c = barycentric_coordinates(p, v0, v1, v2)
            if a >= 0 and b >= 0 and c >= 0:
                z = get_z_coord(p, v0, v1, v2)
                # проверка z-buffer
                if check_z_buffer(x, y, z):
                    pt = a * vt0 + b * vt1 + c * vt2
                    i = w - 1 - int(np.ceil(pt[1] * w))
                    j = int(np.ceil(pt[0] * h))
                    image[x, y, :3] = base * texture[i, j, :3]
    return image


def back_face_culling(p, v0, v1, v2):
    x_min = min(v0[0], v1[0], v2[0])
    A0, A1, A2 = v0, v1, v2
    if v0[0] == x_min:
        A0, A1, A2 = v2, v0, v1
    elif v2[0] == x_min:
        A0, A1, A2 = v1, v2, v0
    r = np.cross((A1 - A0), (A1 - A2))
    if np.any(r <= 0):
        A0, A1 = A1, A0
    c = (v0 - p)
    n = np.cross((A1 - A0), (A2 - A1))
    result = np.dot(c, n)
    return result / 500 + 0.5


def check_z_buffer(x, y, z):
    if z <= z_buffer[x, y]:
        z_buffer[x, y] = z
        return 1
    return 0


def create_image(heigh, widht, background_color):
    image = np.zeros((heigh, widht, 3), np.uint8)
    image[:, :, :] = background_color
    return image


def show_image(image):
    image = np.flipud(image)
    plt.figure()
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    image = np.flipud(image)
    imsave(name, image)

width = 480
height = 480
z_buffer = np.ones((width, height))
z_buffer *= 10

def main():
    # читаем инф-ю из файла obj и записываем списки
    v = []
    vt = []
    vn = []
    f = []
    file = open('3d Model/duck.obj', 'r')
    # file = open('test.obj', 'r')
    line = file.readline()
    while not line[0] == 'v':
        line = file.readline()
    while line[0:2] == 'v ':
        v.append(list(map(float, line[2:].split())))
        line = file.readline()
    while line[0:2] == 'vt':
        vt.append(list(map(float, line[3:].split())))
        line = file.readline()
    while line[0:2] == 'vn':
        vn.append(list(map(float, line[3:].split())))
        line = file.readline()
    while not line[0] == 'f':
        line = file.readline()
    while not line == '':
        line = [j.split(sep='/') for j in line[2:].split()]
        f.append(list((list(i - 1 for i in map(int, line[j])) for j in range(len(line)))))
        line = file.readline()
    file.close()

    # пребразуем списки в матрицы
    v = np.array(v).T
    vn = np.array(vn).T
    vt = np.array(vt).T

    # считываем текстуру
    texture = mpimg.imread('3d Model/duck.png')
    texture_w = texture.shape[0]
    texture_h = texture.shape[1]

    # нахожим матрицу перевода координат верщин и нормалей в мировую систему
    # по условию:
    # alpha_x = 8
    # alpha_y = 12
    # alpha_z = 16
    alpha_x = 0
    alpha_y = 0
    alpha_z = 45
    vec = np.array([-1, 0, -1])
    a = b = c = 0.9
    Mo2w = o2wMatr(alpha_x, alpha_y, alpha_z, vec, a, b, c)

    # получаем координаты вершин в новой системе координат
    v_proj = to_proj_coords(v)
    v = (to_cart_coords(Mo2w @ v_proj))

    # получаем координаты нормалей в новой системе координат
    vn_proj = to_proj_coords(vn)
    vn = (to_cart_coords(linalg.inv(Mo2w.T) @ vn_proj))

    # находим матрицу вычисления координат в системе, связанной с камерой
    # расположение камеры по условию:
    A = np.array([2, 2, 2])
    B = np.array([-2, -2, 0])
    Mw2c = w2cMatr(A[0], A[1], A[2], B[0], B[1], B[2])

    # получаем координаты вершин в новой системе координат
    v_proj = to_proj_coords(v)
    v = (to_cart_coords(Mw2c @ v_proj))

    # получаем координаты нормалей в новой системе координат
    vn_proj = to_proj_coords(vn)
    vn = (to_cart_coords(linalg.inv(Mw2c.T) @ vn_proj))

    # вычисляем матрицу проекции (перспектиную или ортографическую)
    left = min(v[0]) - 0.1
    right = max(v[0]) + 0.1
    bottom = min(v[1]) - 0.1
    top = max(v[1]) + 0.1
    near = min(-v[2]) - 0.1
    far = max(-v[2]) + 0.1
    Mproj = orthProjMatr(left, right, bottom, top, near, far)
    # Mproj = persProjMatr(left, right, bottom, top, near, far)

    # получаем координаты вершин в новой системе координат
    v_proj = to_proj_coords(v)
    v = (to_cart_coords(Mproj @ v_proj))

    # получаем координаты нормалей в новой системе координат
    vn_proj = to_proj_coords(vn)
    vn = (to_cart_coords(linalg.inv(Mproj.T) @ vn_proj))


    # вычисляем матрицу перехода в систему координат области вывода
    x = 0
    y = 0
    Mviewport = viewportMatr(x, y, width, height)

    # получаем координаты вершин в новой системе координат
    v_proj = to_proj_coords(v)
    v = (to_cart_coords(Mviewport @ v_proj))

    # получаем координаты нормалей в новой системе координат
    vn_proj = to_proj_coords(vn)
    vn = (to_cart_coords(linalg.inv(Mviewport.T) @ vn_proj))


    # создаем массив изображения
    image0 = create_image(width, height, np.array((255, 255, 255), dtype=np.uint8))
    image1 = create_image(width, height, np.array((255, 255, 255), dtype=np.uint8))
    image2 = create_image(width, height, np.array((130, 230, 0), dtype=np.uint8))

    count = len(f)
    color = np.array((0, 0, 0), dtype=np.uint8)
    P = np.array([width / 2, height / 2, 3])

    # рендеринг проволочной модели
    for i in range(count):
        v0 = np.array(v[:2, f[i][0][0]])
        v1 = np.array(v[:2, f[i][1][0]])
        v2 = np.array(v[:2, f[i][2][0]])
        v3 = np.array(v[:2, f[i][3][0]])

        image0 = draw_line(image0, v0[0], v0[1], v1[0], v1[1], width, height, color)
        image0 = draw_line(image0, v1[0], v1[1], v2[0], v2[1], width, height, color)
        image0 = draw_line(image0, v2[0], v2[1], v3[0], v3[1], width, height, color)
        image0 = draw_line(image0, v3[0], v3[1], v0[0], v0[1], width, height, color)
        image0 = draw_line(image0, v0[0], v0[1], v2[0], v2[1], width, height, color)

        if i % 100 == 0:
            print(round(i / (count // 100)), '%', sep='', end='\n')
    print('Rendering is done!')

    show_image(image0)
    save_image(image0, 'duck0.jpg')

    # рендеринг модели с гранями
    for i in range(count):
        v0 = np.array(v[:, f[i][0][0]])
        v1 = np.array(v[:, f[i][1][0]])
        v2 = np.array(v[:, f[i][2][0]])
        v3 = np.array(v[:, f[i][3][0]])

        c = back_face_culling(P, v0, v1, v2)
        if c >= 0:
            c *= 255
            if c > 255:
                c = 255
            color = np.array([c, c, c], dtype=np.uint8)
            image1 = draw_face(image1, v0, v1, v2, color)
            image1 = draw_face(image1, v2, v3, v0, color)

        if i % 100 == 0:
            print(round(i / (count // 100)), '%', sep='', end='\n')
    print('Rendering is done!')

    show_image(image1)
    save_image(image1, 'duck1.jpg')

    # рендеринг модели с текстурой
    for i in range(count):
        v0 = np.array(v[:, f[i][0][0]])
        v1 = np.array(v[:, f[i][1][0]])
        v2 = np.array(v[:, f[i][2][0]])
        v3 = np.array(v[:, f[i][3][0]])

        vt0 = np.array(vt[:, f[i][0][1]])
        vt1 = np.array(vt[:, f[i][1][1]])
        vt2 = np.array(vt[:, f[i][2][1]])
        vt3 = np.array(vt[:, f[i][3][1]])

        c = back_face_culling(P, v0, v1, v2)
        if c >= 0:
            c *= 255
            if c > 255:
                c = 255
            image2 = draw_face_texture(image2, v0, v1, v2, vt0, vt1, vt2, texture, texture_w, texture_h, c)
            image2 = draw_face_texture(image2, v2, v3, v0, vt2, vt3, vt0, texture, texture_w, texture_h, c)

        if i % 100 == 0:
            print(round(i / (count // 100)), '%', sep='', end='\n')
    print('Rendering is done!')

    show_image(image2)
    save_image(image2, 'duck2.jpg')


if __name__ == '__main__':
    main()
