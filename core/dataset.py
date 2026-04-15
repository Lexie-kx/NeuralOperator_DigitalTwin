import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# ==========================================
# 针对 CFDBench Cylinder (prop.zip) 的双变量归一化边界
# ==========================================
# 1. 密度 (Density) 边界
DENSITY_MIN = 0.1
DENSITY_MAX = 500.0

# 2. 运动粘度 (Viscosity) 边界
VISCOSITY_MIN = 0.0001
VISCOSITY_MAX = 0.01


class PodDeepONetDataset(Dataset):
    def __init__(self, data_dir, pod_bases_path=None):
        self.data_dir = data_dir
        self.cases = sorted([d for d in os.listdir(data_dir) if d.startswith('case')])

        self.true_coefs = None
        if pod_bases_path and os.path.exists(pod_bases_path):
            pod_data = np.load(pod_bases_path)
            self.true_coefs = pod_data['train_coefs']
            print(f"✅ 成功加载真实 POD 系数，模态数量: {self.true_coefs.shape[1]}")

    def __len__(self):
        return len(self.cases)

    def normalize(self, val, v_min, v_max):
        """通用线性归一化到 [0, 1] 区间"""
        return (val - v_min) / (v_max - v_min)

    def __getitem__(self, idx):
        case_folder = self.cases[idx]
        json_path = os.path.join(self.data_dir, case_folder, 'case.json')

        # 1. 提取双物理参数
        with open(json_path, 'r', encoding='utf-8') as f:
            case_info = json.load(f)
            density = case_info['density']
            viscosity = case_info['viscosity']

        # 2. 分别进行归一化
        den_norm = self.normalize(density, DENSITY_MIN, DENSITY_MAX)
        vis_norm = self.normalize(viscosity, VISCOSITY_MIN, VISCOSITY_MAX)

        # 3. 组合为 2D 向量作为 Branch Net 的输入！
        branch_in = torch.tensor([den_norm, vis_norm], dtype=torch.float32)

        if self.true_coefs is not None:
            target_coefs = torch.tensor(self.true_coefs[idx], dtype=torch.float32)
            return branch_in, target_coefs
        else:
            return branch_in