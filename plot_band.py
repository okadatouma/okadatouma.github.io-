import numpy as np
import matplotlib.pyplot as plt
from data_bravis import BRAVAIS_IC as BRAVAIS_SC
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm




'''軌道を考慮してうまく行列にできたと思っていた関数をここで供養
###hr.datを読み取って、行列にする関数をここで定義
def read_hr_dat_okada(file_path):
    
    #hr.datをここで読み取る
    with open(file_path, "r") as f:
        data = f.readlines()
        
    #ワニエの情報をここで保持しておく
    num_wann = int(data[1].strip())
    nrpts    = int(data[2].strip())
    n_deg_lines = (nrpts + 14) // 15   #degenecyが何行続くのかをここで計算している（一応自動化したい）
    degeneracy = data[3:3+(n_deg_lines)]
    
    #後で縮退度で割るので、ここで縮退度の値は配列でおいておく
    deg_vals = []   
    for line in degeneracy :
        deg_vals.extend(int(x) for x in line.split())    #ここではappendではだめ
    deg_vals = np.array(deg_vals[:nrpts], dtype = int )
    
    #Rについての情報もここでもっておく
    R_list = []
    
    #行列の値をここで配列にして行列化する
    data     = data[3+n_deg_lines:]
    data_    = []
    for idx, line in enumerate( data ) :
        parts = line.split()
        if len(parts) < 7:  # インデックス範囲チェック
            continue
        R = tuple(int(x) for x in parts[0:3])
        
        if idx % (num_wann**2) == 0 :
            R_list.append(R)
            
        col4_Re = float(parts[5])
        col5_Im = float(parts[6])
        data_.append( col4_Re + 1j* col5_Im )
    R_list = np.array(R_list, dtype =int )
    
    #ここでnumpy配列を使う
    arr = np.array(data_)
    matrix = arr.reshape(-1, num_wann, num_wann, order="F")    #Rごとの行列を作成する
    matrix = matrix.transpose(0, 2, 1)                         #(0:i,R:j)の行列にする(本当に正しかったのか要検証)
    matrix = matrix / deg_vals[:, None, None]       #ここで各値を縮退度でわっておく(縮退度での割り方がこれであってるのかが分からない)
    
    
    
    return matrix, num_wann, nrpts, deg_vals, R_list
'''
##2代目行列化する関数
def read_hr_dat_okada(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
        
    #基本情報(ワニエ数、Rの数、縮退度の数)をここで取得
    num_wann = int(data[1].strip())
    nrpts    = int(data[2].strip())
    n_deg_lines = (nrpts + 14) // 15
    degeneracy_lines = data[3:3+n_deg_lines]

    #縮退度をnumpyとして保持しておく 
    deg_vals = []
    for line in degeneracy_lines:
        deg_vals.extend(int(x) for x in line.split())
    deg_vals = np.array(deg_vals[:nrpts], dtype=int)

    #ハミルトニアン本体 
    H_lines = data[3+n_deg_lines:]
    R_list = []
    block_size = num_wann**2
    count = 0  # 有効な行だけカウント

    # Rごとの行列 (nrpts, num_wann, num_wann)
    matrix = np.zeros((nrpts, num_wann, num_wann), dtype=complex)

    for line in H_lines:
        parts = line.split()
        if len(parts) < 7:
            continue  # 一応空行などはスキップ

        # R, i, j
        R = tuple(int(x) for x in parts[0:3])   #別にintのままでいいかも
        i = int(parts[3]) - 1  # 0始まりに直す
        j = int(parts[4]) - 1

        #ブロックの先頭なら R を記録
        if count % block_size == 0:
            R_list.append(R)

        re = float(parts[5])
        im = float(parts[6])

        R_index = count // block_size    # 0,1,2,...,nrpts-1
        val = (re + 1j*im) / deg_vals[R_index]

        matrix[R_index, i, j] = val

        count += 1

    R_list = np.array(R_list, dtype=int)

    #出力は前と同じような形にしておく
    return matrix, num_wann, nrpts, deg_vals, R_list





###作成したmatrixをフーリエ変換する関数
def R_to_k_fourier_okada( matrix , R_list , k_path, num_wann) :
    n_k = len(k_path)
    #フーリエ変換の位相部分をまず最初に計算する
    phase = R_list @ k_path.T 
    exp = np.exp(1j*2*np.pi*(phase))   #(nrpts,n_k)の行列
    
    '''
    ライブラリ使った方がいいのかもしれない
    もし遅かったら以下のようにライブラリを使用
    '''
    #各k点にフーリエ変換する
    H_k = np.tensordot (exp.T, matrix, axes=(1, 0))   #(nrpts,n_k).T @ (nrpts,24,24)=(n_k,24,24)の計算をしている
    
    #各k点において対角化して固有値と固有値ベクトルを配列に入れる
    vals =  np.empty((n_k, num_wann))
    eig_vec =  np.empty((n_k, num_wann, num_wann), dtype=complex)
    orbit_group = np.empty((n_k, num_wann, num_wann), dtype=float)    #それぞれの軌道の関する重みをここで格納する(np.aggange(num_wann))
    for i in range(n_k) :
        eig , vec = np.linalg.eigh(H_k[i])
        vals[i, :] = eig
        eig_vec[i, :, :] = vec
        
        #軌道の重みの情報（ファットバンド、投影DOSを書く用）   
        weight = np.abs(vec)**2   
        orbit_group[i, :, :] = weight    #(n_k,wann)の行列に固有ベクトルを格納
        
    return vals, eig_vec, orbit_group




###対称点からk_pathを作成する
def generate_kpath_okada(BRAVAIS_SC, bravais, a, b, c_over_a = None, alpha = None):
    data = BRAVAIS_SC[bravais]
    if c_over_a is None :
        c= a
    else :
        c = a * (c_over_a)    
        
    '''必要なパラメータについてここに書いておく
    if bravais == "tl1" or "tl2" : c/a needed
    if ravais == "hR1" or "hR2" : alpha needed
    
    !!!まだalphaを用いて計算ができるようにはなっていないことに注意!!!
    '''
    #必要な情報を辞書から取り出す
    special_points = data["special_points"]
    path = data["path"]
    nk_per_segment = np.array(data["nk_per_segment"])       #k_pathの区切り方
    kdist_per_segment = np.array(data["kdist_per_segment"]) #対称点間の距離（格子の大きさの異方性を考慮していないので基本使わない）
    
    # 必要なら N を使って、各対称点間の点数をスケーリングしてもよいが、
    # とりあえずは辞書の nk_per_segment をそのまま使う。
    # （N は今は未使用。後で調整してもいい）
    
    kpath_points = [] #ここにk点を収納していく
    
    #ここではk_path状のk点の座標を出力していく
    for i in range(len(path)-1) :
        k_start = np.array(special_points[path[i]], dtype=float)
        k_end   = np.array(special_points[path[i+1]],dtype=float)
        num     = int(nk_per_segment[i])
        
        line=np.linspace(k_start , k_end , num=num , endpoint=False )          #各対称点の座標を取り出して細かく区切る
        kpath_points.append(line)
        
    #最後の点を追加する
    end_point = np.array(special_points[path[-1]], dtype =float)
    kpath_points.append(end_point[None, :])
    
    #最後にnumpy配列にする
    kpath_points = np.vstack(kpath_points)    
        
    #バンド図でプロットする用のk点の座標をここで作成する 
    #辞書の値を使わずにここでk点間の距離を計算する
    dk_frac = np.diff(kpath_points, axis=0)       
    
    #格子の大きさを考慮して変換する(格子の大きさ1)
    dk_cart = np.empty_like(dk_frac)
    dk_cart[:, 0] = dk_frac[:, 0] / a
    dk_cart[:, 1] = dk_frac[:, 1] / b
    dk_cart[:, 2] = dk_frac[:, 2] / c
    
    #各ステップの距離
    step_len = np.linalg.norm(dk_cart, axis=1)

    # 累積和をとって横軸にする
    k_axis = np.concatenate([[0.0], np.cumsum(step_len)])

    # ついでに、各特異点のインデックスも返しておくと
    # gnuplot/ matplotlib で目盛りを打つときに便利
    special_indices = [0]
    idx = 0
    for n_seg in nk_per_segment:
        idx += int(n_seg)
        special_indices.append(idx)

    return kpath_points, k_axis, special_indices , path
        

#軌道についての辞書
groups = {"Ni_d_x2-y2":[3], 
          "Ni_d_z2"   :[0],
          "Ni_d_xy,yz,zx" :[1, 2, 4],
          "O_p" :[5, 6, 7, 8, 9, 10],
          "La_d":[11, 12, 13, 14, 15],
          "La_f":[16, 17, 18, 19, 20, 21, 22],
          "s"   :[23]
          }



###グループごとの重みを作る関数
def make_group_weight_okada(orbit_group, groups):
    group_weight = {}
    for name, id in groups.items():
        id = np.array(id, dtype=int)
        group_weight[name] = orbit_group[:, id, :].sum(axis=1)
    
    return group_weight

###バンド図をプロットする関数
def plot_bands_okada(vals, group_weight, groups, k_axis, special_indices, path, base_color="0.7", scale=15.0):
    #基本設定情報
    fig, ax = plt.subplots(figsize=(12, 8))
    color_list = ["r","c","y","pink","b","k","g"]
    n_k, num_wann = vals.shape
    
    #axisの規格化
    end=k_axis[-1]
    if end != 0:  # ゼロ除算チェック
        k_axis = k_axis*(6.31464/end)
    
    for n in range(num_wann):    #n_kの数と一致しているはず
        x = k_axis
        y = vals[:, n]
        # まず細いベースのバンド線（どの軌道成分でも共通の線）をかく
        ax.plot(x, y, color=base_color, lw=0.5, zorder=1)
    
    
    # 軌道グループ（辞書でまとめた）ごとに色を変えてプロット 
    for (name, color) in zip(groups.keys(), color_list):
        '''
        if name != "O_p":
            continue
        '''
        gw = group_weight[name]    #(n_k, num_wann)
        
        # ★ legend 用のダミー点（空データ）を先に登録
        ax.scatter([], [], color=color, s=200, label=name)    
           
        for n in range(num_wann):
            x = k_axis
            y = vals[:, n]
            w =gw[:, n]      
            
            if w.max() <=0:
                continue
            
            s= (w / w.max() )* scale
            
            ax.scatter(x, y, s=s, color=color, alpha=0.7, zorder=2 )
       
    # 軸ラベル、タイトル、目盛りなどの設定 
    ax.set_xticks([k_axis[i] for i in special_indices])
    ax.set_xticklabels(path, fontsize=20)    
        
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Fat band")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # 描画範囲（エネルギー）を設定：-4 eV 〜 4 eV
    fermi_energy = 14.0316
    ax.set_ylim(fermi_energy-4.0, fermi_energy+4.0)
    ax.set_xlim(0.0, k_axis[-1])
    # 参考のために E=0 の横線を描画
    ax.axhline(fermi_energy, color='k', linewidth=1.5, linestyle='--', zorder=0)
    ax.legend(fontsize=14, loc='upper right')

    plt.savefig("fatband_s.png", dpi=300)
    plt.show()
###特定の軌道の重みをカラープロットにしたい
def plot_bands_okada2(vals, group_weight, groups, k_axis, special_indices, path,
                     target_orbit=None, base_color="0.7", scale=30.0):

    fig, ax = plt.subplots(figsize=(12, 8))
    n_k, num_wann = vals.shape
    
    # k 軸の規格化
    end = k_axis[-1]
    if end != 0:
        k_axis = k_axis * (6.31464/end)

    # ベースのバンド線
    for n in range(num_wann):
        ax.plot(k_axis, vals[:, n], color=base_color, lw=0.5, zorder=1)

    # ========（重要）特定の軌道グループのみ cmap で可視化する ========
    if target_orbit is not None:
        gw = group_weight[target_orbit]   # (n_k, num_wann)

        # 重みの正規化
        wmax = gw.max()
        if wmax <= 0:
            wmax = 1.0

        norm = Normalize(vmin=0, vmax=wmax)
        cmap = cm.get_cmap("plasma")  # 好みで変更可

        # legend のためのダミー
        ax.scatter([], [], c=[], cmap=cmap, label=target_orbit)

        # 実際の scatter
        for n in range(num_wann):
            x = k_axis
            y = vals[:, n]
            w = gw[:, n]

            if w.max() <= 0:
                continue

            s = (w / w.max()) * scale

            ax.scatter(
                x, y,
                c=w, cmap=cmap, norm=norm,
                s=s,
                edgecolors="none",
                alpha=0.9,
                zorder=3
            )

        # カラーバー追加
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f"Weight: {target_orbit}")

    # 軸ラベルなど
    ax.set_xticks([k_axis[i] for i in special_indices])
    ax.set_xticklabels(path, fontsize=18)

    fermi_energy = 14.0316
    ax.set_ylim(fermi_energy - 4, fermi_energy + 4)
    ax.axhline(fermi_energy, color="k", linestyle="--", lw=1.2)

    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Fat band — color = weight({target_orbit})")
    ax.legend()

    plt.savefig("fatband_color_target.png", dpi=300)
    plt.show()
    
###Dosをプロットする関数
def plot_dos_okada(vals, group_weight, groups, fermi_energy=14.016, energy_range=(-4.0, 4.0), bin_width=0.05):
    # エネルギー範囲とビンの設定
    energy_min = fermi_energy + energy_range[0]
    energy_max = fermi_energy + energy_range[1]
    bins = np.arange(energy_min, energy_max + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 全体のDOSを計算
    #all_dos, _ = np.histogram(vals.flatten(), bins=bins)
    
    #規格化
    bin_all_width = _[1]-[0]
    aLL = len(vals.flatten())
    all_dos = all_dos/(aLL * bin_all_width)

    # プロットの準備
    fig, ax = plt.subplots(figsize=(10, 6))

    # 全体のDOSをプロット
    #ax.plot(bin_centers, all_dos, label='Total DOS', color='black', linewidth=1.5)

    

    # 各軌道グループごとのDOSを計算してプロット
    #軌道ごとにプロットする場合は、以下のように計算する
    for name in groups.keys():
        if not (  name =="O_p" ) :
            continue
        
        gw = group_weight[name]  #(n_k, num_wann)
        group_vals = []
        n_k, num_wann = gw.shape
        for i in range(n_k):
            for j in range(num_wann):
                weight = gw[i, j]
                if weight > 1e-6:  # 非常に小さい重みは無視
                    group_vals.extend([vals[i, j]] * int(weight * 100))  # 重みに応じて値を追加

        if len(group_vals) == 0:
            continue

        group_dos, _ = np.histogram(group_vals, bins=bins)    #group_dos[i](iに入ったデータ点の個数)、_[i]、_[i+1]binの左右の端
        #規格化
        bin_width = _[1]-_[0]
        N=len(group_vals)
        group_dos = group_dos/(N*bin_width)
        
        ax.plot(bin_centers, group_dos, label=f'{name} DOS', linewidth=1.2)

    # 軸ラベル、タイトル、目盛りなどの設定
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Density of States (states/eV)')
    ax.set_title('Projected Density of States')
    ax.legend(fontsize=12)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    plt.savefig("projected_only_O_p_dos.png", dpi=300)
    plt.show()
'''
###フェルミ面の3次元プロット
def plot_fermi_surfaces_okada(matrix, R_list, num_wann,
                              band_indices, fermi_energy,
                              nk=(20,20,20),
                              frac_range=((0,1),(0,1),(0,1)),
                              colors=None):
    """
    複数バンドのフェルミ面を同時に3Dプロットする関数。

    Parameters
    ----------
    band_indices : list or tuple
        フェルミ面を表示したいバンド番号のリスト  (例: [4,5,6])
    colors : list of colors (optional)
        バンドごとに色を指定 (例: ["r","g","b"])
    """

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    if colors is None:
        # バンド数に応じて自動で色を割り当てる
        import matplotlib.cm as cm
        cmap = cm.get_cmap("hsv", len(band_indices))
        colors = [cmap(i) for i in range(len(band_indices))]

    # --- 1. k メッシュを生成 ---
    (kx_min,kx_max), (ky_min,ky_max), (kz_min,kz_max) = frac_range
    Nkx, Nky, Nkz = nk
    kx = np.linspace(kx_min, kx_max, Nkx)
    ky = np.linspace(ky_min, ky_max, Nky)
    kz = np.linspace(kz_min, kz_max, Nkz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kpoints = np.stack([KX,KY,KZ], axis=-1)
    kpoints_flat = kpoints.reshape(-1,3)
    Nk_tot = len(kpoints_flat)

    # --- 2. H(k) を全k点で構築 ---
    phase = R_list @ kpoints_flat.T
    exp_factor = np.exp(1j * 2.0 * np.pi * phase)
    H_k = np.tensordot(exp_factor.T, matrix, axes=(1,0))

    # --- 3. 固有値を全部計算 ---
    eigvals_all = np.linalg.eigvalsh(H_k)  # shape = (Nk_tot, num_wann)
    eigvals_grid = eigvals_all.reshape(Nkx, Nky, Nkz, num_wann)

    # --- 4. プロット ---
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")

    for idx, band in enumerate(band_indices):
        E_grid = eigvals_grid[:,:,:,band] - fermi_energy

        volume = np.transpose(E_grid, (2,1,0))  # (z,y,x)

        verts, faces, normals, values = marching_cubes(volume,
                                                       level=0.0,
                                                       spacing=(kz[1]-kz[0],
                                                                ky[1]-ky[0],
                                                                kx[1]-kx[0]))
        verts_xyz = verts[:, ::-1]  # (x,y,z)

        mesh = Poly3DCollection(verts_xyz[faces],
                                alpha=0.65,
                                facecolor=colors[idx],
                                edgecolor="none")
        ax.add_collection3d(mesh)
        ax.plot([], [], label=f"band {band}", color=colors[idx])

    ax.set_xlabel("k_x (frac.)")
    ax.set_ylabel("k_y (frac.)")
    ax.set_zlabel("k_z (frac.)")
    ax.set_title("Fermi Surfaces")

    ax.set_xlim(kx_min, kx_max)
    ax.set_ylim(ky_min, ky_max)
    ax.set_zlim(kz_min, kz_max)

    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return fig, ax
'''
