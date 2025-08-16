#!/usr/bin/env python3
# 简化的动态数据处理脚本
import os
import sys
import numpy as np
import torch
import igl

def process_dynamic_scene(scene_path, numsamples=50000, dim=3, num_timesteps=4):
    """
    处理单个动态场景的时序OBJ文件
    
    Args:
        scene_path: 场景目录路径，应包含 mesh_z_up_t0.obj ~ mesh_z_up_t3.obj
        numsamples: 每个时刻的采样点数
        dim: 空间维度
        num_timesteps: 时刻数量
    """
    print(f"处理动态场景: {scene_path}")
    
    # 检查时序OBJ文件是否存在
    missing_files = []
    for t in range(num_timesteps):
        mesh_file = os.path.join(scene_path, f"mesh_z_up_t{t}.obj")
        if not os.path.exists(mesh_file):
            missing_files.append(f"mesh_z_up_t{t}.obj")
    
    if missing_files:
        print(f"❌ 缺少以下文件: {missing_files}")
        return False
    
    # 处理每个时刻
    all_sampled_points = []
    all_speeds = []
    obstacle_positions = []
    
    for t in range(num_timesteps):
        mesh_file = os.path.join(scene_path, f"mesh_z_up_t{t}.obj")
        print(f"  处理时刻 {t}: {mesh_file}")
        
        try:
            # 读取mesh
            v, f = igl.read_triangle_mesh(mesh_file)
            print(f"    顶点数: {v.shape[0]}, 面数: {f.shape[0]}")
            
            # 计算障碍物质心（简化版本）
            centroid = np.mean(v, axis=0)
            obstacle_positions.append(centroid)
            print(f"    质心位置: {centroid}")
            
            # 简化的速度场生成（这里使用随机采样作为示例）
            # 在实际使用中，你需要调用完整的采样函数
            sampled_points = (np.random.rand(numsamples, 2*dim) - 0.5)
            speed = np.random.rand(numsamples, 2) * 0.5 + 0.5
            
            all_sampled_points.append(sampled_points)
            all_speeds.append(speed)
            
            # 保存当前时刻的数据
            np.save(os.path.join(scene_path, f'sampled_points_t{t}.npy'), sampled_points)
            np.save(os.path.join(scene_path, f'speed_t{t}.npy'), speed)
            print(f"    ✅ 保存时刻{t}的数据")
            
        except Exception as e:
            print(f"    ❌ 处理时刻{t}失败: {e}")
            return False
    
    # 计算运动向量
    motion_vectors = []
    for t in range(1, num_timesteps):
        motion_vec = obstacle_positions[t] - obstacle_positions[t-1]
        motion_vectors.append(motion_vec)
    
    # 保存运动信息
    motion_info = {
        'positions': obstacle_positions,
        'vectors': motion_vectors
    }
    np.save(os.path.join(scene_path, 'motion_info.npy'), motion_info)
    
    # 生成共享的Fourier特征矩阵
    B = 0.5 * np.random.normal(0, 1, size=(3, 128))
    np.save(os.path.join(scene_path, 'B.npy'), B)
    
    print(f"✅ 场景处理完成!")
    print(f"   - 生成了{num_timesteps}个时刻的数据")
    print(f"   - 每个时刻{numsamples}个采样点")
    print(f"   - 障碍物运动向量: {len(motion_vectors)}个")
    
    return True

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
    else:
        # 默认路径
        scene_path = "./datasets/gibson_3smalltree_dynamic/my_final_scene/0"
    
    print("=== 动态障碍物数据处理 ===")
    print(f"目标路径: {scene_path}")
    
    if not os.path.exists(scene_path):
        print(f"❌ 路径不存在: {scene_path}")
        print("请确保路径正确，并且已经放入时序OBJ文件")
        return
    
    # 处理场景
    success = process_dynamic_scene(scene_path)
    
    if success:
        print("\n🎉 数据处理成功!")
        print("现在可以运行训练脚本:")
        print("python train/train_dynamic.py")
    else:
        print("\n❌ 数据处理失败")
        print("请检查OBJ文件是否正确放置")

if __name__ == "__main__":
    main()