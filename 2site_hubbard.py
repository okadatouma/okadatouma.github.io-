#ここでは練習のためにnサイトハバード模型の厳密対角化を行う。(1次元)
#2次元への拡張は、また後で行う
#tj模型と異なり、2重占有は禁止していない

import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("site_hubbard", "/home/25_okada/dft/sotuken/hubbard/1site_hubbard.py")
site_hubbard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site_hubbard)
make_local_ops = site_hubbard.make_local_ops
make_matrix = site_hubbard.make_matrix
hermite_conjugate = site_hubbard.hermite_conjugate


def projection(h, n):
    from itertools import product
    # occup = [sum(state) for state in product([0, 1], repeat=4)]
    # index = [i for i, occ in enumerate(occup) if occ == n]
    index = [i for i, state in enumerate(product([0, 1], repeat=4)) if sum(state)==n]
    return h[index, :][:, index]


def solve_hubbard_2site(t, U, mu=0, n=None):

    local_ops = make_local_ops()
    cdag = local_ops['c^+']
    I = local_ops['I']
    F = local_ops['F']

    # creation operators
    Cdag = {}
    Cdag['1u'] = make_matrix(I, I, I, cdag)
    Cdag['1d'] = make_matrix(I, I, cdag, F)
    Cdag['2u'] = make_matrix(I, cdag, F, F)
    Cdag['2d'] = make_matrix(cdag, F, F, F)

    C = {}  # annihilation operators
    N = {}  # number operators
    for key, cdag in Cdag.items():
        C[key] = hermite_conjugate(cdag)
        N[key] = cdag @ C[key]

    hamil = 0
    # t
    for key1, key2 in [('1u', '2u'), ('1d', '2d'), ('2u', '1u'), ('2d', '1d')]:
        hamil += -t * Cdag[key1] @ C[key2]
    # U
    for key1, key2 in [('1u', '1d'), ('2u', '2d')]:
        hamil += U * N[key1] @ N[key2]
    # mu
    for n_op in N.values():
        hamil += -mu * n_op
    # print("H =\n", hamil)

    if n is not None:
        # projection to n-particle state
        hamil = projection(hamil, n)

    eigval, eigvec = np.linalg.eigh(hamil)

    return eigval, eigvec


def main():
    t = 1.0
    U = 4.0
    mu = 2

    E, vec = solve_hubbard_2site(t, U, mu)
    # E, vec = solve_hubbard_2site(t, U, mu, n=2)

    print("\nE =\n", E)

    print("\nEigenvectors =")
    for i in range(vec.shape[1]):
        print(vec[:, i])


if __name__ == '__main__':
    main()
