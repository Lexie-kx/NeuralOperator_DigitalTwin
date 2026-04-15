import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from core.model import PodBranchNet


def main():
    pod_bases_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cylinder\pod_bases.npz"
    save_weight_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cylinder\best_pod_deeponet.pth"

    data = np.load(pod_bases_path)
    num_cases = len(data['train_coefs'])

    # 构建归一化输入参数 (模拟不同入口流速 vel_in 归一化到 0~1)
    x_train = torch.linspace(0, 1, num_cases).view(-1, 1)
    y_train = torch.tensor(data['train_coefs'], dtype=torch.float32)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PodBranchNet(input_dim=1, output_dim=y_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("🚀 开始 Phase 2 代理模型快速训练...")
    for epoch in range(1500):  # 圆柱非线性较强，多跑几轮
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 300 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), save_weight_path)
    print(f"✅ Cylinder 模型权重已保存至: {save_weight_path}")


if __name__ == "__main__":
    main()