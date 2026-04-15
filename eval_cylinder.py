import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from core.model import PodBranchNet


def main():
    pod_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cylinder\pod_bases.npz"
    weight_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cylinder\best_pod_deeponet.pth"
    # ⚠️ 专属命名，绝对不会覆盖 Cavity！
    save_img = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\results\cylinder_bc_comparison.png"

    data = np.load(pod_path)

    # 根据你之前图片的比例，推测圆柱网格也是 64x64
    res_x, res_y = 64, 64
    spatial_points = res_x * res_y
    nt = data['mean_u'].size // spatial_points

    model = PodBranchNet(input_dim=1, output_dim=data['train_coefs'].shape[1])
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    # 取中间的一个工况（流速适中，涡街最漂亮）
    test_idx = len(data['train_coefs']) // 2
    test_x = torch.tensor([[test_idx / len(data['train_coefs'])]], dtype=torch.float32)

    with torch.no_grad():
        pred_c = model(test_x).numpy()[0]
    true_c = data['train_coefs'][test_idx]

    u_mean = data['mean_u'].reshape(nt, res_y, res_x)[-1]
    u_basis_all = data['basis_u'].reshape(nt, res_y, res_x, -1)
    u_basis_last = u_basis_all[-1]

    u_true = u_mean + np.dot(u_basis_last, true_c)
    u_pred = u_mean + np.dot(u_basis_last, pred_c)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin, vmax = u_true.min(), u_true.max()

    im0 = axes[0].imshow(u_true, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth (Cylinder)", fontsize=14)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pred, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("DeepONet Surrogate", fontsize=14)
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(u_true - u_pred)
    im2 = axes[2].imshow(err, cmap='hot', origin='lower')
    axes[2].set_title(f"MAE: {np.mean(err):.5f}", fontsize=14)
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_img), exist_ok=True)
    plt.savefig(save_img, bbox_inches='tight', dpi=300)
    print(f"✨ 阶段二浴火重生！卡门涡街图已保存至：{save_img}")
    plt.show()


if __name__ == "__main__":
    main()