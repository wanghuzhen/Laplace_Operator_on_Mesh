from numpy.core.fromnumeric import shape
import scipy.sparse.linalg as lg
from scipy import sparse
import time
from read_off import *

# 全部输出查看数据
# np.set_printoptions(threshold = np.inf)


def norm(a):
    # return np.linalg.norm(a)
    return np.sqrt(np.sum(a**2, axis=1))


def tri_area(p1, p2, p3):
    return norm(np.cross(p1 - p2, p1 - p3)) / 2


def cot(a, b):
    return np.einsum('ij,ij->i', a, b) / norm(np.cross(a, b))


def vertexArea(vertex, tri):
    a = tri_area(vertex[tri[:, 0]], vertex[tri[:, 1]],
                 vertex[tri[:, 2]]).reshape((1, tri.shape[0]))
    j = np.zeros((tri.shape[1], tri.shape[0]))
    # print(j.flatten().shape)
    s = np.concatenate((a, a, a), axis=0) / 3
    # print(s.flatten().shape)
    A = sparse.csr_matrix((s.flatten(), (tri.T.flatten(), j.flatten())),
                          shape=(vertex.shape[0], 1)).T.toarray()
    A = sparse.spdiags(A, [0], vertex.shape[0], vertex.shape[0])
    return A


def cot_weight(faces, vertexs, tri, vertex):
    i = tri.copy()
    j = i.copy()
    i = i.flatten(order='F')
    j[:, [0, 1, 2]] = j[:, [1, 2, 0]]
    j = j.flatten(order='F')
    # print(i)
    # print(j)
    # print('===========')
    a1 = cot(vertex[tri[:, 0]] - vertex[tri[:, 1]],
             vertex[tri[:, 0]] - vertex[tri[:, 2]])
    a2 = cot(vertex[tri[:, 1]] - vertex[tri[:, 0]],
             vertex[tri[:, 1]] - vertex[tri[:, 2]])
    a3 = cot(vertex[tri[:, 2]] - vertex[tri[:, 1]],
             vertex[tri[:, 2]] - vertex[tri[:, 0]])
    a_all = (np.concatenate((a3, a1, a2), axis=0) / 2)
    I = np.concatenate((i, j, i, j), axis=0)
    J = np.concatenate((j, i, i, j), axis=0)
    S = np.concatenate((-a_all, -a_all, a_all, a_all), axis=0)
    # print(I)
    # print('============')
    # print(J)
    # print('============')
    # print(S)
    w = sparse.csr_matrix((S, (I, J)), shape=(vertexs, vertexs))
    # print(w)
    return w


def cot_lp(A, W, k=30):
    eig_value, eig_vec = lg.eigsh(W, M=A, k=30, which='LM', sigma=0)
    if (eig_value[0] < 0 or eig_value[0] < 1E-7):
        eig_value[0] = 0.
    return eig_value, eig_vec


if __name__ == '__main__':
    print('=========================')
    file = './001-2354.off'
    start_read = time.time()
    vertexs, faces, edges, ver, tri = read_off(file)
    end_read = time.time()
    print('Read .off file time is {:.3f} second'.format(end_read - start_read))
    start_cal_A = time.time()
    A = vertexArea(ver, tri)
    end_cal_A = time.time()
    print('Calculate matrix A time is {:.3f} second'.format(end_cal_A -
                                                            start_cal_A))
    # print(A)
    start_cal_W = time.time()
    W = cot_weight(faces, vertexs, tri, ver)
    end_cal_W = time.time()
    print('Calculate matrix W time is {:.3f} second'.format(end_cal_W -
                                                            start_cal_W))
    # print(w)
    # a = A.toarray()
    # w = W.toarray()
    # E = np.dot(lg.inv(A),W).tocsr()
    start_cal_eig = time.time()
    eig_value, eig_vec = cot_lp(A, W, k=10)
    end_cal_eig = time.time()
    print('Calculate eig time is {:.3f} second'.format(end_cal_eig -
                                                       start_cal_eig))
    print('All process time is {:.3f} second'.format(end_cal_eig - start_read))
    print(eig_value)
    print(eig_vec.shape)
