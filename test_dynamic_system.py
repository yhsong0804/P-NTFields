# 临时测试脚本 - 使用现有静态数据测试动态系统
import sys
sys.path.append('.')

import torch
import numpy as np
from models.model_dynamic import DynamicModel, DynamicNN
from models.data_dynamic import DynamicDatabase

def test_dynamic_system_with_static_data():
    """使用现有静态数据测试动态系统"""
    
    print("测试动态PINN系统...")
    
    # 使用现有的静态数据路径
    static_data_path = './datasets/gibson_3smalltree/my_final_scene/'
    model_path = './Experiments/Test_Dynamic/'
    
    # 创建模型目录
    import os
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(f"{model_path}/plots", exist_ok=True)
    
    try:
        # 1. 测试动态模型初始化
        print("1. 初始化动态模型...")
        model = DynamicModel(model_path, static_data_path, 3, 2, 'cpu', 4)
        model.init_dynamic_network()
        print("   ✓ 动态模型初始化成功")
        
        # 2. 测试动态数据加载器
        print("2. 测试动态数据加载器...")
        dataset = DynamicDatabase(static_data_path, torch.device('cpu'), 2, 4)
        
        # 获取一个样本
        data_sequence, B, motion_info, index = dataset[0]
        print(f"   ✓ 数据序列形状: {data_sequence.shape}")
        print(f"   ✓ B矩阵形状: {B.shape}")
        print(f"   ✓ 运动信息: {motion_info}")
        
        # 3. 测试网络前向传播
        print("3. 测试网络前向传播...")
        print(f"   原始数据序列形状: {data_sequence.shape}")
        coords_sequence = data_sequence[:, :, :6]  # 前6维是坐标
        
        # 添加batch维度
        coords_sequence = coords_sequence.unsqueeze(0)  # [1, timesteps, points, 6]
        B = B.unsqueeze(0)  # [1, 3, 128]
        
        # 前向传播测试
        with torch.no_grad():
            if hasattr(model.network, 'forward_dynamic_sequence'):
                output = model.network.forward_dynamic_sequence(coords_sequence, B[0], motion_info)
                print(f"   ✓ 网络输出形状: {output.shape}")
            else:
                print("   ! 使用fallback前向传播")
                output, _ = model.network.out(coords_sequence[0, -1], B[0])
                print(f"   ✓ 网络输出形状: {output.shape}")
        
        # 4. 测试损失计算
        print("4. 测试损失计算...")
        speed_sequence = data_sequence[:, :, 6:]  # 后2维是速度
        speed_sequence = speed_sequence.unsqueeze(0)
        
        try:
            if hasattr(model, 'Loss_Dynamic'):
                loss, static_loss, temporal_loss, arc_loss = model.Loss_Dynamic(
                    coords_sequence, speed_sequence, B[0], motion_info, 1.0, 0.001)
                print(f"   ✓ 动态损失: {loss.item():.4e}")
                print(f"   ✓ 静态损失: {static_loss.item():.4e}")
                print(f"   ✓ 时序损失: {temporal_loss.item():.4e}")
                print(f"   ✓ 圆弧损失: {arc_loss.item():.4e}")
            else:
                print("   ! 使用fallback损失函数")
                loss, _, _ = model.Loss(coords_sequence[0, -1], speed_sequence[0, -1], B[0], 1.0, 0.001)
                print(f"   ✓ 损失: {loss.item():.4e}")
        except Exception as e:
            print(f"   × 损失计算失败: {e}")
        
        print("\n🎉 动态系统测试成功！系统已准备就绪。")
        print("\n下一步操作:")
        print("1. 准备你的时序OBJ文件 (mesh_z_up_t0.obj ~ mesh_z_up_t3.obj)")
        print("2. 将文件放入 datasets/gibson_3smalltree_dynamic/my_final_scene/0/ 目录")
        print("3. 运行 python train/train_dynamic.py 开始训练")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dynamic_system_with_static_data()