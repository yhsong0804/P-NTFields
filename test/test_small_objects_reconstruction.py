import sys
sys.path.append('.')

import numpy as np
import torch
import matplotlib.pyplot as plt
from models import model_res_sigmoid_multi as md
import os

def test_small_objects_reconstruction():
    """测试小障碍物重现效果"""
    
    print("=== 小障碍物重现效果测试 ===")
    
    # 设置模型路径
    modelPath = './Experiments/Gib_multi_small_objects'
    dataPath = './datasets/gibson/'
    
    # 加载训练好的模型
    model = md.Model(modelPath, dataPath, 3, 2, device='cuda:0')
    
    # 寻找最新的模型文件
    model_files = [f for f in os.listdir(modelPath) if f.endswith('.pt')]
    if not model_files:
        print("错误：未找到训练好的模型文件")
        return
    
    latest_model = max(model_files, key=lambda x: int(x.split('_')[2]))
    model_file_path = os.path.join(modelPath, latest_model)
    
    print(f"加载模型: {model_file_path}")
    model.load(model_file_path)
    
    # 创建测试网格
    print("生成测试网格...")
    limit = 0.5
    resolution = 100  # 提高分辨率以更好地捕捉小障碍物
    
    x = np.linspace(-limit, limit, resolution)
    y = np.linspace(-limit, limit, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 设置起点（可以根据您的场景调整）
    start_point = [-0.25, -0.25, 0.0]  # Gibson场景的典型起点
    
    # 创建查询点
    query_points = np.zeros((resolution * resolution, 6))  # 6D: [start_x, start_y, start_z, end_x, end_y, end_z]
    query_points[:, :3] = start_point  # 起点
    query_points[:, 3] = X.flatten()   # 终点x
    query_points[:, 4] = Y.flatten()   # 终点y
    query_points[:, 5] = 0.0           # 终点z（假设2D平面）
    
    # 转换为PyTorch张量
    query_tensor = torch.tensor(query_points, dtype=torch.float32, device='cuda:0')
    
    print("计算速度场...")
    with torch.no_grad():
        # 计算速度场
        speed_values = model.Speed(query_tensor)
        speed_field = speed_values.cpu().numpy().reshape(X.shape)
        
        # 计算旅行时间
        travel_times = model.TravelTimes(query_tensor)
        travel_time_field = travel_times.cpu().numpy().reshape(X.shape)
        
        # 计算tau值
        tau_values = model.Tau(query_tensor)
        tau_field = tau_values.cpu().numpy().reshape(X.shape)
    
    # 可视化结果
    print("生成可视化图像...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 速度场
    im1 = axes[0, 0].imshow(speed_field, extent=[-limit, limit, -limit, limit], 
                           origin='lower', cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title('速度场 (Speed Field)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 旅行时间等高线
    contour = axes[0, 1].contour(X, Y, travel_time_field, levels=20, cmap='bone', linewidths=0.5)
    im2 = axes[0, 1].imshow(speed_field, extent=[-limit, limit, -limit, limit], 
                           origin='lower', cmap='viridis', alpha=0.7, vmin=0, vmax=1)
    axes[0, 1].set_title('速度场 + 旅行时间等高线')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Tau场
    im3 = axes[1, 0].imshow(tau_field.squeeze(), extent=[-limit, limit, -limit, limit], 
                           origin='lower', cmap='plasma', vmin=0, vmax=1)
    axes[1, 0].set_title('Tau 场')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 旅行时间场
    im4 = axes[1, 1].imshow(travel_time_field, extent=[-limit, limit, -limit, limit], 
                           origin='lower', cmap='coolwarm')
    axes[1, 1].set_title('旅行时间场')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, 'small_objects_reconstruction_test.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"测试结果保存到: {output_file}")
    
    # 分析小障碍物检测效果
    print("\n=== 小障碍物检测分析 ===")
    
    # 统计低速度区域（可能对应障碍物）
    low_speed_threshold = 0.3
    low_speed_mask = speed_field < low_speed_threshold
    low_speed_percentage = np.sum(low_speed_mask) / speed_field.size * 100
    
    print(f"低速度区域 (< {low_speed_threshold}) 占比: {low_speed_percentage:.2f}%")
    
    # 检测速度场的局部最小值（可能对应小障碍物）
    from scipy import ndimage
    local_minima = ndimage.minimum_filter(speed_field, size=5) == speed_field
    very_low_speed = speed_field < 0.1
    small_obstacle_candidates = local_minima & very_low_speed
    
    num_small_obstacles = np.sum(small_obstacle_candidates)
    print(f"检测到可能的小障碍物数量: {num_small_obstacles}")
    
    # 计算速度场的变化梯度
    grad_x, grad_y = np.gradient(speed_field)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    high_gradient_areas = gradient_magnitude > np.percentile(gradient_magnitude, 95)
    
    print(f"高梯度区域 (可能的障碍物边界) 占比: {np.sum(high_gradient_areas)/high_gradient_areas.size*100:.2f}%")
    
    # 输出统计信息
    print(f"\n速度场统计:")
    print(f"  最小值: {speed_field.min():.4f}")
    print(f"  最大值: {speed_field.max():.4f}")
    print(f"  均值: {speed_field.mean():.4f}")
    print(f"  标准差: {speed_field.std():.4f}")
    
    plt.show()
    
    return speed_field, travel_time_field, tau_field

if __name__ == "__main__":
    test_small_objects_reconstruction()