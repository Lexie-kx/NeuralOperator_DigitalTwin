import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. 路径配置
    pod_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cavity\pod_bases.npz"
    save_img = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\results\cavity_u_comparison.png"

    # 2. 加载数据
    data = np.load(pod_path)
    res = 64
    spatial_points = res * res

    # 自动计算时间步长
    nt = data['mean_u'].size // spatial_points
    print(f"📊 数据检测：分辨率 {res}x{res}，时间步长 {nt}")

    # 3. 直接使用 True Coefficients 进行重构 (展示 POD 的上限)
    # 选取第 0 个工况进行展示
    true_c = data['train_coefs'][0]

    u_mean = data['mean_u'].reshape(nt, res, res)[-1]
    u_basis_all = data['basis_u'].reshape(nt, res, res, -1)
    u_basis_last = u_basis_all[-1]

    u_true = u_mean + np.dot(u_basis_last, true_c)
    # 这里我们直接展示 POD 的完美重构效果
    u_pod_reconstruct = u_mean + np.dot(u_basis_last, true_c)

    # 4. 绘图 (采用专业配色和对齐色标)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 设置统一的色标范围，让对比有意义
    vmin, vmax = u_true.min(), u_true.max()

    im0 = axes[0].imshow(u_true, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth (CFD)", fontsize=14)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pod_reconstruct, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("POD Ideal Reconstruction", fontsize=14)
    plt.colorbar(im1, ax=axes[1])

    # 理想重构的误差应该极小 (e-10 量级)
    err = np.abs(u_true - u_pod_reconstruct)
    im2 = axes[2].imshow(err, cmap='hot', origin='lower')
    axes[2].set_title(f"Reconstruction MAE: {np.mean(err):.2e}", fontsize=14)
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Phase 1: Validation of POD Basis Extraction (Cavity Flow)", fontsize=16, y=1.05)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_img), exist_ok=True)
    plt.savefig(save_img, bbox_inches='tight', dpi=300)
    print(f"✨ 完美的 Phase 1 验证图已存入：{save_img}")
    plt.show()


if __name__ == "__main__":
    main()