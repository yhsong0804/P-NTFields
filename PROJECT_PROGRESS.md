# 动态障碍物PINN项目 - 完整记录

## 项目概述
将静态gibson环境的PINN扩展为支持动态障碍物的版本，通过输入4个时刻(t, t+1, t+2, t+3)的OBJ文件来表征障碍物移动，最终生成向前圆弧状的速度场和时间场。

## 已完成的工作

### 1. 代码架构设计
- **多时刻输入架构**: 同时处理4个时刻的场景数据
- **时序注意力机制**: 使用MultiheadAttention聚合时序信息
- **动态损失函数**: 包含时序一致性和圆弧约束
- **运动感知编码**: 提取障碍物运动信息并融入网络

### 2. 创建的核心文件

#### 数据处理模块
- `dataprocessing/speed_sampling_dynamic.py`: 时序数据采样处理
- `models/data_dynamic.py`: 动态数据加载器

#### 网络架构
- `models/model_dynamic.py`: 
  - DynamicNN类: 支持时序输入的神经网络
  - TemporalAttention类: 时序注意力模块
  - DynamicModel类: 动态训练模型

#### 训练脚本
- `train/train_dynamic.py`: 动态训练主程序
- `test_dynamic_setup.py`: 数据准备检查工具
- `test_dynamic_system.py`: 系统完整性测试

### 3. 关键技术改进

#### 网络结构修改
```python
# 原始输入: 6维坐标 (起点3维 + 终点3维)
# 动态输入: 6维坐标 + 64维时间编码 + 64维运动特征
# 总维度: 6 + 64 + 64 = 134维
```

#### 时序注意力机制
```python
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4):
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads)
        self.time_embedding = nn.Linear(1, feature_dim)
```

#### 动态损失函数
```python
total_loss = (static_loss + 
              0.1 * temporal_consistency_loss +
              0.05 * arc_constraint_loss)
```

### 4. 数据集结构设计
```
datasets/gibson_3smalltree_dynamic/my_final_scene/
├── 0/
│   ├── mesh_z_up_t0.obj  # 时刻0的场景
│   ├── mesh_z_up_t1.obj  # 时刻1的场景(障碍物移动)
│   ├── mesh_z_up_t2.obj  # 时刻2的场景
│   ├── mesh_z_up_t3.obj  # 时刻3的场景
│   ├── sampled_points_t0.npy ~ sampled_points_t3.npy
│   ├── speed_t0.npy ~ speed_t3.npy
│   └── B.npy
└── 1/ (同样结构)
```

## 运行流程

### 步骤1: 准备时序OBJ数据
- 创建4个时刻的OBJ文件，表示障碍物逐渐移动
- 放入对应目录: `datasets/gibson_3smalltree_dynamic/my_final_scene/0/`

### 步骤2: 生成训练数据
```bash
python dataprocessing/speed_sampling_dynamic.py
```

### 步骤3: 测试系统
```bash
python test_dynamic_setup.py      # 检查数据准备
python test_dynamic_system.py     # 测试系统完整性
```

### 步骤4: 开始训练
```bash
python train/train_dynamic.py
```

## 技术细节

### 维度匹配解决方案
原始模型在`model_res_sigmoid_multi.py:159`行有维度问题：
```python
# 原来: self.encoder.append(Linear(2*h_size,h_size))  # 256->128
# 修改: self.encoder[0] = Linear(2*128 + 64, 128)     # 320->128 (增加运动特征)
```

### 圆弧轨迹生成机制
1. **障碍物运动提取**: 从时序mesh计算运动向量
2. **圆弧方向计算**: 生成垂直于运动方向的圆弧切线
3. **距离衰减**: 越接近障碍物，圆弧效应越强
4. **动态修正**: 在速度场中添加圆弧分量

### 预期效果
- 在动态障碍物位置显示向前的圆弧状速度场
- 体现障碍物运动趋势的预测性避障
- 比静态版本更平滑和智能的轨迹规划

## 下一步工作
1. 准备4个时刻的OBJ文件 (mesh_z_up_t0.obj ~ mesh_z_up_t3.obj)
2. 运行数据预处理生成时序采样数据
3. 开始动态PINN训练
4. 分析生成的速度场和时间场效果

## 故障排除

### 常见问题
1. **维度不匹配**: 检查encoder第一层输入维度设置
2. **设备错误**: 确保所有tensor在同一设备(CPU/GPU)
3. **内存不足**: 减少batch_size或采样点数量
4. **数据缺失**: 使用静态数据fallback机制

### 调试工具
- `test_dynamic_setup.py`: 检查文件完整性
- `test_dynamic_system.py`: 验证网络前向传播
- 训练过程中的loss监控和可视化

## 联系方式
如果连接中断，可以通过以下关键词快速恢复上下文：
- "动态障碍物PINN"
- "时序注意力机制" 
- "圆弧状速度场"
- "gibson_3smalltree_dynamic"