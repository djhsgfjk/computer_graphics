import math

def edge_len(v1, v2):
    return math.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2 + (v2[2] - v1[2]) ** 2)


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

    # считаем сумму длин всех ребер
    edges_sum = 0
    for e in edges:
        edges_sum += edge_len(vertices[e[0]], vertices[e[1]])

    # находим вершину, которая принадлежит максимальному кол-ву граней
    f_max = 0
    v_num = []
    for i in range(v_amount):
        n = len(v_fs[i])
        if n > f_max:
            f_max = n
    for i in range(v_amount):
        n = len(v_fs[i])
        if n == f_max:
            v_num.append(i)

    # выводим результат
    print(edges_sum)
    for i in range(len(v_num)):
        print(vertices[v_num[i]], f_max)


if __name__ == '__main__':
    main()
