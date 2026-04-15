import os
import numpy as np

# ==========================================
# 物理常量预设 (针对 64x64 网格)
# ==========================================
NX, NY = 64, 64
TIME_STEPS = 1000
DX = 0.22 / (NX - 1)  # X轴步长
DY = 0.12 / (NY - 1)  # Y轴步长


def compute_pod(fluc_matrix, component_name, energy_threshold=0.999):
    print(f"[{component_name}] 正在进行 SVD 分解 (矩阵大小: {fluc_matrix.shape})...")
    U, S, Vh = np.linalg.svd(fluc_matrix, full_matrices=False)

    energy = S ** 2
    cum_energy = np.cumsum(energy) / np.sum(energy)
    num_modes = np.searchsorted(cum_energy, energy_threshold) + 1

    basis = U[:, :num_modes]
    coefs = np.diag(S[:num_modes]) @ Vh[:num_modes, :]

    print(f"  -> 截取模态数: {num_modes}")
    print(f"  -> 累积能量占比: {cum_energy[num_modes - 1] * 100:.4f}%\n")

    return basis, coefs.T, num_modes


def calc_spatial_gradient(flat_array, axis, d_step):
    """
    计算展开向量的空间梯度
    axis=1 对应 Y轴 (dy)，axis=2 对应 X轴 (dx)
    """
    # 将一维向量重塑为 (Time, Y, X)
    reshaped = flat_array.reshape(TIME_STEPS, NY, NX)
    # 计算有限差分梯度
    grad = np.gradient(reshaped, d_step, axis=axis)
    return grad.flatten()


def main():
    data_dir = r"D:\prop"
    weights_dir = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\prop"
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, 'pod_bases.npz')

    cases = sorted([d for d in os.listdir(data_dir) if d.startswith('case')])
    print(f"🚀 开启 PINO 物理基因提取，共发现 {len(cases)} 个工况...")

    U_snaps, V_snaps = [], []
    for case in cases:
        case_path = os.path.join(data_dir, case)
        U_snaps.append(np.load(os.path.join(case_path, 'u.npy')).flatten())
        V_snaps.append(np.load(os.path.join(case_path, 'v.npy')).flatten())

    U_snaps = np.array(U_snaps).T
    V_snaps = np.array(V_snaps).T

    # 1. 提取均值场及其偏导数
    mean_u = np.mean(U_snaps, axis=1)
    mean_v = np.mean(V_snaps, axis=1)

    print("📐 正在计算均值场的空间偏导数...")
    # 对 U 求 X 偏导 (axis=2)，对 V 求 Y 偏导 (axis=1)
    mean_u_dx = calc_spatial_gradient(mean_u, axis=2, d_step=DX)
    mean_v_dy = calc_spatial_gradient(mean_v, axis=1, d_step=DY)

    # 2. 雷诺分解
    U_fluc = U_snaps - mean_u[:, None]
    V_fluc = V_snaps - mean_v[:, None]

    # 3. 提取 POD 基底
    basis_u, coefs_u, k_u = compute_pod(U_fluc, "U-Velocity")
    basis_v, coefs_v, k_v = compute_pod(V_fluc, "V-Velocity")

    # 4. 提取基底的偏导数 (PINO 核心操作)
    print("📐 正在对 POD 基底进行有限差分预求导...")
    basis_u_dx = np.zeros_like(basis_u)
    basis_v_dy = np.zeros_like(basis_v)

    for i in range(k_u):
        basis_u_dx[:, i] = calc_spatial_gradient(basis_u[:, i], axis=2, d_step=DX)
    for i in range(k_v):
        basis_v_dy[:, i] = calc_spatial_gradient(basis_v[:, i], axis=1, d_step=DY)

    train_coefs = np.concatenate([coefs_u, coefs_v], axis=1)
    total_modes = k_u + k_v

    print(f"📦 正在打包物理基因与导数矩阵至: {save_path}")
    np.savez(save_path,
             basis_u=basis_u, mean_u=mean_u, basis_u_dx=basis_u_dx, mean_u_dx=mean_u_dx,
             basis_v=basis_v, mean_v=mean_v, basis_v_dy=basis_v_dy, mean_v_dy=mean_v_dy,
             train_coefs=train_coefs)

    print(f"✅ 物理提取完毕！网络输出维度依然保持为: {total_modes}")


if __name__ == "__main__":
    main()