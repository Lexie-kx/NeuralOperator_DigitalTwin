import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from core.dataset import PodDeepONetDataset
from core.model import PodBranchNet


def main():
    # ==========================================
    # 1. 核心路径配置
    # ==========================================
    data_dir = r"D:\prop"
    pod_bases_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\prop\pod_bases.npz"
    save_weight_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\prop\best_pino_deeponet.pth"

    # ==========================================
    # 2. 实例化数据集与加载器
    # ==========================================
    print("正在加载 prop 数据集...")
    dataset = PodDeepONetDataset(data_dir=data_dir, pod_bases_path=pod_bases_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ==========================================
    # 3. 预加载 PINO 导数矩阵
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用计算设备: {device}")

    print("📐 正在将物理导数矩阵加载至显存...")
    pod_data = np.load(pod_bases_path)

    basis_u_dx_T = torch.tensor(pod_data['basis_u_dx'].T, dtype=torch.float32).to(device)
    mean_u_dx = torch.tensor(pod_data['mean_u_dx'], dtype=torch.float32).to(device)

    basis_v_dy_T = torch.tensor(pod_data['basis_v_dy'].T, dtype=torch.float32).to(device)
    mean_v_dy = torch.tensor(pod_data['mean_v_dy'], dtype=torch.float32).to(device)

    # ==========================================
    # 4. 实例化模型与优化器
    # ==========================================
    model = PodBranchNet(input_dim=2, output_dim=43).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6)

    # ==========================================
    # 5. 开启 PINO 饱和式训练
    # ==========================================
    epochs = 3000
    best_loss = float('inf')

    # ✅ 核心修改点：将物理权重从 1.0 暴砍到 0.01
    # 破解梯度打架问题，让网络优先保证流场长相对 (L_Data)，顺带满足质量守恒 (L_PDE)
    lambda_pde = 0.01

    print(f"\n🚀 PINO 引擎点火 (当前物理权重 lambda_pde={lambda_pde})，开启受物理定律约束的回归训练...")
    for epoch in range(epochs):
        model.train()
        total_data_loss = 0.0
        total_pde_loss = 0.0
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            preds = model(batch_x)

            loss_data = criterion(preds, batch_y)

            pred_u = preds[:, :23]
            pred_v = preds[:, 23:]

            dudx = mean_u_dx + torch.matmul(pred_u, basis_u_dx_T)
            dvdy = mean_v_dy + torch.matmul(pred_v, basis_v_dy_T)

            continuity_residual = dudx + dvdy
            loss_pde = torch.mean(continuity_residual ** 2)

            loss = loss_data + lambda_pde * loss_pde

            loss.backward()
            optimizer.step()

            total_data_loss += loss_data.item()
            total_pde_loss += loss_pde.item()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_data_loss = total_data_loss / len(train_loader)
        avg_pde_loss = total_pde_loss / len(train_loader)

        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | L_Total: {avg_loss:.4f} | L_Data: {avg_data_loss:.4f} | L_PDE: {avg_pde_loss:.4f} | LR: {current_lr:.6e}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_weight_path)
                print(f"   [+] 破纪录！最佳 PINO 模型已保存 -> {save_weight_path}")

    print(f"\n🎉 训练完成！全局最低 Total Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()