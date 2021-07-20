import numpy as np


def read_off(file_path):
    with open(file_path, 'r') as f:
        lst = f.readlines()
        vertexs, faces, edges = list(
            map(lambda x: int(x), lst[1].strip('\n').split(' ')))
        # print(vertexs,faces,edges)
        ver = np.array(
            list(
                map(
                    lambda x:
                    [float(i) for i in x.strip('\n').split(' ')[:-1]],
                    lst[2:2 + vertexs])))
        tri = np.array(
            list(
                map(lambda x: [int(i) for i in x.strip('\n').split(' ')[1:]],
                    lst[2 + vertexs:])))
        return vertexs, faces, edges, ver, tri


if __name__ == '__main__':
    file = r'./001-2354.off'
    vertexs, faces, edges, ver, tri = read_off(file)
    # print(len(tri)==faces)
    # print(len(ver)==vertexs)
