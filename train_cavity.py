import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from core.model import PodBranchNet


def main():
    pod_bases_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cavity\pod_bases.npz"
    save_weight_path = r"D:\Research_Projects\DeepONet-SurrogatePythonProject\weights\cavity\best_pod_deeponet.pth"

    # 1. 加载数据
    data = np.load(pod_bases_path)
    # 假设输入是工况索引的归一化 (0~1)
    x_train = torch.linspace(0, 1, len(data['train_coefs'])).view(-1, 1)
    y_train = torch.tensor(data['train_coefs'], dtype=torch.float32)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. 实例化模型 (1个输入 -> k个输出)
    model = PodBranchNet(input_dim=1, output_dim=y_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. 快速训练
    print("🚀 开始 Phase 1 快速训练...")
    for epoch in range(1000):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), save_weight_path)
    print(f"✅ Cavity 模型已保存至: {save_weight_path}")


if __name__ == "__main__":
    main()