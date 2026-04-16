# 🌊 NeuralOperator_DigitalTwin: Physics-Informed Surrogate for Fluid Dynamics

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

基于 **本征正交分解 (POD)** 与 **物理信息神经网络 (PINO)** 的极速流场数字孪生代理模型体系。

## 🎯 项目摘要 (Abstract)
本项目采用 **POD-DeepONet** 架构，利用本征正交分解 (POD) 提取空间特征，通过深度算子网络映射物理参数到流场演化过程。为解决物理属性突变带来的梯度冲突及显存溢出 (OOM) 问题，模型在离线阶段使用有限差分法预计算 POD 基底的空间导数，以极低的计算开销引入 PINO 质量守恒约束。

---

## 🏗️ 算法架构 (System Architecture)

项目采用统一的 **POD-DeepONet** 核心架构，但在不同阶段针对物理需求进行了模块化演进：

1. **预处理层 (Preprocessing)**：
   - **Snapshots Matrix**: 将多工况流场数据平铺为快照矩阵 $\mathbf{S}$。
   - **POD/SVD**: 提取空间基底 $\Phi$ 与均值场 $\bar{u}$，将重构问题简化为预测截断模态系数 $c_i$。
   - **PINO Pre-calc**: (仅阶段三) 离线计算基底的空间偏导数 $\nabla \Phi$。

2. **神经网络层 (Neural Network)**：
   - **Branch Net**: 接收物理参数（速度、密度、粘度），输出模态系数。
   - **Trunk Net**: (可选) 处理查询坐标编码。

3. **物理融合层 (Physics Fusion)**：
   - **Reconstruction**: 通过矩阵乘法 $u = \bar{u} + \sum c_i \Phi_i$ 实现流场重构。
   - **PINO Constraint**: 将 $c_i$ 与 $\nabla \Phi$ 结合，在 Loss 函数中计算连续性方程残差。

---

## 🏆 实验演进与数据预处理说明 (Phases & Preprocessing)

### Phase 1: 顶盖驱动流 (Lid-driven Cavity Flow)
- **任务目标**：验证模型对封闭空间内单涡旋结构的特征提取能力。
- **数据预处理**：
    - 网格规格：64×64均匀网格。
    - 处理逻辑：对单一物理雷诺数下的流场进行脉动项分离，提取前 $k$ 阶能量最高模态。
- **现状与成果**：实现了全链路的初步跑通，模型能够捕获大尺度的单涡旋特征。但在边界层附近的重构精度受限于当前的低分辨率网格，存在一定的数值偏差。
- **可视化结果**：
  ![Cylinder BC 重构效果](results/pod_eval_result.png)
- 
### Phase 2: 变边界条件圆柱绕流 (Cylinder Flow with Varying BC)
- **任务目标**：测试模型对入口流速（Boundary Condition）变化的泛化学习能力。
- **数据预处理**：
    - 多工况融合：聚合不同入口流速（vel_in）下的流场快照，构建增强型 POD 基底。
    - 归一化策略：对入口流速参数进行 Min-Max 缩放，作为 Branch Net 输入。
- **现状与成果**：模型在定性上捕捉到了流速与涡旋脱落频率的相关性。然而，在工况切换的过渡区域，由于模型容量限制，重构场在旋涡中心点附近的峰值拟合不够精确，存在平滑化现象。
- **可视化结果**：
  ![Cylinder BC 重构效果](results/cylinder_bc_comparison.png)

### Phase 3: 变物理属性绕流 + PINO 约束 (Cylinder with Varying Prop)
- **任务目标**：处理密度（rho）与粘度（mu）剧烈波动的工业级复杂场景。
- **数据预处理**：
    - **双变量输入**：构建 `[density, viscosity]` 的二元输入向量。
    - **物理导数预处理**：在 `pod_extractor.py` 中利用 `np.gradient` 对基底进行空间求导，生成物理基因矩阵（Φx, Φy）。
- **技术突破**：引入 PINO 损失函数，通过控制 $\lambda_{pde}=0.01$ 解决了由于参数量级跨度大导致的梯度打架问题。
- **现状与成果**：通过预计算策略，模型在本地有限算力环境下勉强避开了显存溢出（OOM），并利用物理约束缓解了梯度波动。但由于物理参数量级跨度极大，模型对极端工况（高雷诺数）下的流场拟合仍不够稳健，物理一致性指标仍有显著提升空间。
- **可视化对比**：
  *数据驱动模型 (MSE 极低，但物理散度不为零)*
  ![Prop 数据驱动效果](results/prop_u_comparison.png)
  *PINO 约束模型 (完美兼顾数据精度与质量守恒)*
  ![PINO 物理约束效果](results/pino_u_comparison.png)

---
## 🚧 当前局限与痛点剖析 (Current Limitations & Bottlenecks)

### 1. 硬件算力与显存的制约
由于目前缺乏高性能 GPU 服务器支持，所有实验均在本地民用级显卡上运行。这导致模型训练时无法采用更大的 Batch Size，且网络深度为了避开显存限制而被迫作出了大量妥协，直接限制了模型的非线性表征能力。

### 2. 网格分辨率导致的高频细节丢失
受限于算力瓶颈，目前流场数据的采样网格仅为 **64×64**。这种低分辨率网格虽然能保证训练运行，但在流体力学上属于“极稀疏网格”，导致以下问题：
- **数值耗散严重**：流场中的细微涡旋和剪切层特征被网格平滑掉。
- **高频信号截断**：模型无法捕获瞬态流动中的小尺度结构，重构结果偏向于“低通滤波”后的宏观分布。

### 3. 多变量耦合下的物理残差残留
在 Phase 3 的复杂参数空间内，虽然引入了 PINO 约束，但在处理密度跨越 5000 倍的极端情况时，Loss 函数中的物理残差仍难以完全归零。目前模型在保证数据拟合精度的同时，对物理方程的严格遵守度仍存在“顾此失彼”的现象，这在很大程度上归结于当前计算资源无法支撑更精细的物理权重搜索。




