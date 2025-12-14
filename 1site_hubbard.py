import numpy as np

#1次元1サイトのハバード模型の厳密対角化を行う
#演算子の定義を行う関数
def make_local_ops():
    #演算子を辞書として、ここで保持する
    ops = {}

    # c^+の定義を行う
    cdag = np.zeros((2,2))
    cdag[1, 0] = 1
    ops['c^+'] = cdag

    # cの定義を行う
    c = cdag.transpose()
    ops['c'] = c

    # I
    ops['I'] = np.identity(2)

    # F (for fermionic anticommutation)
    ops['F'] = np.diag((1.0, -1.0))

    return ops

#上で作成した演算子の辞書を入力変数として、行列を作成する関数
def make_matrix(*ops):
    r = 1.0
    for op in ops[::-1]:
        r = np.kron(op, r)
    return r

#行列の複素共役をとる関数
def hermite_conjugate(mat):
    return mat.conj().T

#作成したハミルトニアンを厳密対角化で解く関数
def solve_hubbard_1site(U, mu=0):

    local_ops = make_local_ops()
    cdag = local_ops['c^+']
    I = local_ops['I']
    F = local_ops['F']

    # creation operators
    Cdag = {}
    Cdag['1u'] = make_matrix(I, cdag)  # F represents fermionic anticommutation
    Cdag['1d'] = make_matrix(cdag, F)

    C = {}  # annihilation operators
    N = {}  # number operators
    for key, cdag in Cdag.items():
        C[key] = hermite_conjugate(cdag)
        N[key] = cdag @ C[key]

    hamil = U * N['1u'] @ N['1d'] - mu * (N['1u'] + N['1d'])
    print("H =\n", hamil)

    eigval, eigvec = np.linalg.eigh(hamil)

    return eigval, eigvec

#作成した関数を使って、ここで実際の計算を行う
def main():
    U = 10.0
    mu = 2

    E, vec = solve_hubbard_1site(U, mu)

    print("\nE =\n", E)

    print("\nEigenvectors =")
    for i in range(vec.shape[1]):
        print(vec[:, i])


if __name__ == '__main__':
    main()
