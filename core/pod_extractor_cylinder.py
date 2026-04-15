import os
import numpy as np

# ==========================================
# ⚠️ 请确认你的圆柱绕流数据集路径！
# ==========================================
DATA_DIR = r"D:\bc"  # 指向你的阶段二圆柱数据
WEIGHTS_DIR = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cylinder"


def compute_pod(fluc_matrix, energy_threshold=0.999):
    print(f"正在进行全局 SVD 分解 (矩阵大小: {fluc_matrix.shape})...")
    U, S, Vh = np.linalg.svd(fluc_matrix, full_matrices=False)
    cum_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    num_modes = np.searchsorted(cum_energy, energy_threshold) + 1
    return U[:, :num_modes], S[:num_modes], num_modes


def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误：未找到数据路径 -> {DATA_DIR}")
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, 'pod_bases.npz')

    print(f"🔍 正在深度扫描 {DATA_DIR}...")
    all_u_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower() == 'u.npy':
                all_u_files.append(os.path.join(root, file))

    all_u_files = sorted(all_u_files)
    if not all_u_files:
        print("❌ 未发现任何 u.npy 文件。")
        return

    print(f"🚀 发现 {len(all_u_files)} 个工况，正在进行对齐加载...")
    snaps = []
    target_size = None

    for i, f_path in enumerate(all_u_files):
        try:
            u_data = np.load(f_path).flatten()
            if target_size is None:
                target_size = u_data.size
            if u_data.size == target_size:
                snaps.append(u_data)
            elif u_data.size > target_size:
                snaps.append(u_data[:target_size])
        except Exception as e:
            pass

    snaps = np.array(snaps).T
    print(f"📦 矩阵构建完成，形状: {snaps.shape}")

    mean_u = np.mean(snaps, axis=1)
    fluc_u = snaps - mean_u[:, None]
    basis_u, S_u, k = compute_pod(fluc_u)
    train_coefs = (basis_u.T @ fluc_u).T

    np.savez(save_path, basis_u=basis_u, mean_u=mean_u, train_coefs=train_coefs)
    print(f"✅ Phase 2 基底提取成功！提取模态数: {k}")


if __name__ == "__main__":
    main()