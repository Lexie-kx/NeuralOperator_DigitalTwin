import os
import numpy as np

# ==========================================
# Phase 1 专用配置
# ==========================================
DATA_DIR = r"D:\cavity"
WEIGHTS_DIR = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cavity"


def compute_pod(fluc_matrix, energy_threshold=0.999):
    print(f"正在进行 SVD 分解 (矩阵大小: {fluc_matrix.shape})...")
    U, S, Vh = np.linalg.svd(fluc_matrix, full_matrices=False)
    cum_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    num_modes = np.searchsorted(cum_energy, energy_threshold) + 1
    return U[:, :num_modes], S[:num_modes], num_modes


def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误：路径不存在 -> {DATA_DIR}")
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, 'pod_bases.npz')

    # 1. 扫描文件
    print(f"🔍 正在深度扫描 {DATA_DIR}...")
    all_u_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower() == 'u.npy':
                all_u_files.append(os.path.join(root, file))

    all_u_files = sorted(all_u_files)
    if len(all_u_files) == 0:
        print("❌ 未发现数据文件。")
        return

    # 2. 数据对齐加载
    print(f"🚀 发现 {len(all_u_files)} 个文件，正在进行形状对齐加载...")
    snaps = []
    target_size = None  # 以第一个文件的总大小为基准

    for i, f_path in enumerate(all_u_files):
        try:
            u_data = np.load(f_path).flatten()

            # 初始化基准大小
            if target_size is None:
                target_size = u_data.size
                print(f"📏 设定基准数据长度: {target_size}")

            # 形状对齐检查
            if u_data.size == target_size:
                snaps.append(u_data)
            elif u_data.size > target_size:
                # 如果数据多了，截断处理
                snaps.append(u_data[:target_size])
                print(f"✂️  文件 {i} 数据过长，已截断对齐")
            else:
                # 如果数据少了，只能跳过，否则矩阵会报错
                print(f"⚠️  跳过文件 {i} ({os.path.basename(os.path.dirname(f_path))}): 数据量不足")

        except Exception as e:
            print(f"❌ 读取失败 {f_path}: {e}")

    # 3. 构建矩阵
    if len(snaps) < 2:
        print("❌ 剩余可用工况不足以进行 POD 分解！")
        return

    snaps = np.array(snaps).T  # 这次绝对不会再报 ValueError 了
    print(f"📦 矩阵对齐完成，最终形状: {snaps.shape}")

    # 4. 计算 POD
    mean_u = np.mean(snaps, axis=1)
    fluc_u = snaps - mean_u[:, None]
    basis_u, S_u, k = compute_pod(fluc_u)

    # 投影获取系数
    train_coefs = (basis_u.T @ fluc_u).T

    # 5. 保存
    np.savez(save_path,
             basis_u=basis_u,
             mean_u=mean_u,
             train_coefs=train_coefs)

    print(f"✅ Phase 1 提取圆满成功！")
    print(f"   - 提取模态数: {k}")
    print(f"   - 保存路径: {save_path}")


if __name__ == "__main__":
    main()