# 动态障碍物的速度场采样
import os
import sys
import numpy as np
import torch
import igl

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataprocessing.speed_sampling_gpu as static_sampler

def sample_dynamic_speed_sequence(base_path, numsamples, dim, num_timesteps=4):
    """
    为移动障碍物场景生成时序速度场数据
    
    Args:
        base_path: 场景文件夹路径
        numsamples: 每个时刻的采样点数
        dim: 空间维度 (通常为3)
        num_timesteps: 时刻数量
    
    Returns:
        序列化的采样点和速度场数据
    """
    
    # 存储所有时刻的数据
    all_sampled_points = []
    all_speeds = []
    
    # 为每个时刻单独采样
    for t in range(num_timesteps):
        mesh_file = f"{base_path}/mesh_z_up_t{t}.obj"
        
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Missing mesh file: {mesh_file}")
        
        print(f"Processing time step {t}: {mesh_file}")
        
        # 为当前时刻生成采样点
        sampled_points_t, speed_t = sample_timestep_speed(
            mesh_file, numsamples, dim, base_path)
        
        all_sampled_points.append(sampled_points_t)
        all_speeds.append(speed_t)
        
        # 保存当前时刻的数据
        np.save(f'{base_path}/sampled_points_t{t}.npy', sampled_points_t)
        np.save(f'{base_path}/speed_t{t}.npy', speed_t)
    
    # 生成共享的Fourier特征矩阵
    B = 0.5 * np.random.normal(0, 1, size=(3, 128))
    np.save(f'{base_path}/B.npy', B)
    
    return all_sampled_points, all_speeds, B

def sample_timestep_speed(mesh_file, numsamples, dim, output_path):
    """
    为单个时刻的mesh采样速度场
    """
    # 读取mesh
    v, f = igl.read_triangle_mesh(mesh_file)
    
    # 设置采样参数
    limit = 0.5
    margin = limit/12.0  # 边界带宽度
    offset = margin/10.0  # 内层宽度
    
    # 调用原有的静态采样函数
    sampled_points, speed = static_sampler.point_rand_sample_bound_points(
        numsamples, dim, v, f, offset, margin)
    
    return sampled_points, speed

def extract_obstacle_motion(base_path, num_timesteps=4):
    """
    从时序mesh中提取障碍物的运动信息
    用于后续的动态约束和圆弧轨迹生成
    """
    obstacle_positions = []
    
    for t in range(num_timesteps):
        mesh_file = f"{base_path}/mesh_z_up_t{t}.obj"
        v, f = igl.read_triangle_mesh(mesh_file)
        
        # 简单的障碍物质心提取（可以根据实际情况优化）
        # 假设移动的障碍物在mesh的特定区域
        obstacle_centroid = extract_moving_obstacle_centroid(v, f)
        obstacle_positions.append(obstacle_centroid)
    
    # 计算障碍物运动速度和方向
    motion_vectors = []
    for t in range(1, num_timesteps):
        motion_vec = obstacle_positions[t] - obstacle_positions[t-1]
        motion_vectors.append(motion_vec)
    
    return obstacle_positions, motion_vectors

def extract_moving_obstacle_centroid(vertices, faces):
    """
    提取移动障碍物的质心位置
    这里需要根据你的具体场景来识别哪部分是移动障碍物
    """
    # 简化版本：假设mesh中心区域是移动障碍物
    # 实际使用时可能需要更复杂的逻辑来识别移动部分
    center = np.mean(vertices, axis=0)
    
    # 可以基于距离、体积或其他几何特征来识别移动障碍物
    # 这里返回整体质心作为示例
    return center

# 批处理函数
def process_dynamic_dataset(dataset_root, numsamples_per_timestep=50000):
    """
    批量处理动态数据集中的所有场景
    """
    scene_dirs = [d for d in os.listdir(dataset_root) 
                  if os.path.isdir(os.path.join(dataset_root, d))]
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(dataset_root, scene_dir)
        print(f"Processing dynamic scene: {scene_path}")
        
        try:
            sample_dynamic_speed_sequence(
                scene_path, numsamples_per_timestep, dim=3)
            print(f"Successfully processed {scene_path}")
        except Exception as e:
            print(f"Error processing {scene_path}: {e}")

if __name__ == "__main__":
    # 示例用法
    dataset_root = "./datasets/gibson_3smalltree_dynamic"
    process_dynamic_dataset(dataset_root)