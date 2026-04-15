import torch
import torch.nn as nn


class PodBranchNet(nn.Module):
    """
    针对 prop.zip (双变量物理属性) 的升级版极速回归网络
    输入: 密度 + 粘度 (2D)
    输出: U 和 V 的综合主模态系数 (43D)
    """

    def __init__(self, input_dim=2, output_dim=43):
        super(PodBranchNet, self).__init__()

        # 考虑到输入变量变成了 2 个，非线性映射难度增加
        # 网络架构保持不变，但两端的端口已完美适配 prop 数据集
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    dummy_input = torch.randn(16, 2)  # 模拟 BatchSize=16, 2个输入变量
    model = PodBranchNet(input_dim=2, output_dim=43)
    output = model(dummy_input)
    print(f"✅ 模型输入维度: {dummy_input.shape}")
    print(f"✅ 模型输出维度: {output.shape} (必须是 [batch_size, 43])")