#ここでは、1次元mサイトのハバード模型の厳密対角化を行うコード
#コードの設計は、1site,2siteと同じ

#2次元への拡張は、また後で行う
#tj模型と異なり、2重占有は禁止していない

#1siteで作成した関数はここでimportしておく
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("site_hubbard", "/home/25_okada/dft/sotuken/hubbard/1site_hubbard.py")
site_hubbard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site_hubbard)
make_local_ops = site_hubbard.make_local_ops
make_matrix = site_hubbard.make_matrix
hermite_conjugate = site_hubbard.hermite_conjugate

#全ヒルベルト空間のハミルトニアンから、粒子数nの部分空間だけを抜き出した部分行列を返す関数
#ハミルトニアンはNと可換なので、粒子数でブロック対角化して、射影する
#変数は、mサイト、粒子数n
def projection(h, n, m):
    from itertools import product
    index = [i for i, state in enumerate(product([0, 1], repeat=2*m)) if sum(state)==n]
    return h[index, :][:, index]
"""
itertools.productでは、多重ループを生成する。[0,1]を4回重ねたループを生成していて、
0,(0,0,0,0)
1,(0,0,0,1)
・・
15,(1,1,1,1)のようになる
"""

#ここでハミルトニアンを作成する
def solve_hubbard_msite( m, t, U, mu=0, n=None):

    local_ops = make_local_ops()
    cdag = local_ops['c^+']
    I = local_ops['I']
    F = local_ops['F']

    # mサイトの生成消滅演算子をここで作成する
    # 1,2サイトとは異なり、リスト、辞書ではなくnumpyを使うとはやいか
    #indexは、1u,1d,2u,2d,・・mu,mdとする
    L = 2*m
    dim = 2**L
    Cdag = [None]*L
    
    for i in range(L) : 
        pos = (L-1) - i 
        ops_list = [I]*L
        ops_list[pos] = cdag
        for j in range(pos+1,L):
            ops_list[j]= F
            
        Cdag[i] = make_matrix(*ops_list)
        
    C = [hermite_conjugate(op) for op in Cdag]
    N = [Cdag[p] @ C[p] for p in range(L)]



    hamil = np.zeros((dim, dim), dtype=np.complex128)
    
    # ホッピング項tを作成
    #nearest-neighborのみを考慮する（必要に応じてsecond,thirdを入れていく）
    for i in range (m-1):
        hamil += -t * Cdag[2*i] @ C[2*(i+1)] -t*Cdag[2*(i+1)]@C[2*i]
        hamil += -t * Cdag[(2*i)+1] @ C[2*(i+1)+1] -t*Cdag[2*(i+1)+1]@C[(2*i)+1]
        
    # ハバードU項を作成
    for i in range(m):
        hamil += U * N[2*i] @ N[2*i+1]
        
    # muの項を作成
    for n_op in N :
        hamil += -mu * n_op
        
    # print("H =\n", hamil)

    if n is not None:
        # projection to n-particle state
        hamil = projection(hamil, n , m)

    eigval, eigvec = np.linalg.eigh(hamil)

    return eigval, eigvec

#実際の厳密対角化計算はここで行う
def main():
    t = 0.0
    U = 10.0
    mu = 2
    m=  1

    E, vec = solve_hubbard_msite(m, t, U, mu, n = None)
    

    print("\nE =\n", E)

    print("\nEigenvectors =")
    for i in range(vec.shape[1]):
        print(vec[:, i])


if __name__ == '__main__':
    main()
