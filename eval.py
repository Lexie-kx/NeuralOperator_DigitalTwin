import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.dataset import PodDeepONetDataset
from core.model import PodBranchNet


def main():
    # ==========================================
    # 1. 路径与模型配置 (已对齐 PINO 专版)
    # ==========================================
    data_dir = r"D:\prop"
    pod_bases_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\prop\pod_bases.npz"

    # ⚠️ 核心修正：明确指向刚刚训练出的 PINO 权重文件！
    model_weight_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\prop\best_pino_deeponet.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    dataset = PodDeepONetDataset(data_dir=data_dir, pod_bases_path=pod_bases_path)

    # 网络架构必须匹配：2 进 (密度, 粘度)，43 出
    model = PodBranchNet(input_dim=2, output_dim=43).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # ==========================================
    # 2. 抽取测试样本进行推断
    # ==========================================
    test_idx = 10  # 抽取第 10 个工况进行展示
    test_input, true_coefs = dataset[test_idx]

    with torch.no_grad():
        pred_coefs = model(test_input.unsqueeze(0).to(device)).cpu().numpy()[0]
    true_coefs = true_coefs.numpy()

    print("\n✅ PINO 模型推断完成！")
    print(f"输入参数 [密度(归一化), 粘度(归一化)]: {test_input.numpy()}")
    print("-" * 50)
    print(f"预测系数 (前5个): {pred_coefs[:5]}")
    print(f"真实系数 (前5个): {true_coefs[:5]}")

    coef_mae = np.mean(np.abs(pred_coefs - true_coefs))
    print("-" * 50)
    print(f"🎯 系数平均绝对误差 (MAE): {coef_mae:.4f}")

    # ==========================================
    # 3. 加载物理基因进行 U 场重构
    # ==========================================
    print("\n正在进行全尺寸流场重构...")
    pod_data = np.load(pod_bases_path)

    # U 方向有 23 个模态，截取前 23 个系数
    pred_coefs_u = pred_coefs[:23]
    true_coefs_u = true_coefs[:23]

    basis_u = pod_data['basis_u']  # Shape: (4096000, 23)
    mean_u = pod_data['mean_u']  # Shape: (4096000,)

    pred_flow_u = mean_u + np.dot(basis_u, pred_coefs_u)
    true_flow_u = mean_u + np.dot(basis_u, true_coefs_u)

    print(f"✅ U 场重构成功，准备渲染图像...")

    # ==========================================
    # 4. 可视化 (自动适应超高分辨率数据)
    # ==========================================
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 自动推断时空矩阵并提取最后稳定的一帧
        if len(true_flow_u) == 1000 * 64 * 64:
            H, W = 64, 64
            plot_true = true_flow_u.reshape(1000, H, W)[-1]
            plot_pred = pred_flow_u.reshape(1000, H, W)[-1]
        else:
            side = int(np.sqrt(len(true_flow_u)))
            plot_true = true_flow_u.reshape(side, -1)
            plot_pred = pred_flow_u.reshape(side, -1)

        im0 = axes[0].imshow(plot_true, cmap='jet', origin='lower')
        axes[0].set_title("Ground Truth (U)")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(plot_pred, cmap='jet', origin='lower')
        axes[1].set_title("PINO Reconstruction (U)")
        plt.colorbar(im1, ax=axes[1])

        error_map = np.abs(plot_true - plot_pred)
        im2 = axes[2].imshow(error_map, cmap='hot', origin='lower')
        # 这里把误差平均值打在标题上，方便你一眼看出精度
        axes[2].set_title(f"Absolute Error (Mean: {np.mean(error_map):.4f})")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        save_path = "results/pino_u_comparison.png"
        plt.savefig(save_path)
        print(f"🚀 PINO 高保真重构图已保存至: {os.path.abspath(save_path)}")
        plt.show()

    except Exception as e:
        print(f"\n⚠️ 绘图中断: {e}")


if __name__ == "__main__":
    main()