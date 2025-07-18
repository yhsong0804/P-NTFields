# verify_samples.py
import open3d as o3d
import numpy as np
import os

# --- 您需要修改的配置 ---
# 1. 设置您要检查的场景路径
SCENE_PATH = './datasets/gibson_3smalltree/0/' # 检查场景0

# 2. 设置要加载的障碍物模型文件名 (归一化后的 .off 文件)
OBSTACLE_FILE = 'mesh_z_up_scaled.off'

# 3. 设置要加载的采样点文件名
SAMPLED_POINTS_FILE = 'sampled_points.npy'
# --- 配置结束 ---


def visualize():
    print("--- 开始加载数据... ---")
    
    # 1. 加载障碍物模型
    mesh_path = os.path.join(SCENE_PATH, OBSTACLE_FILE)
    if not os.path.exists(mesh_path):
        print(f"错误: 找不到障碍物文件: {mesh_path}")
        return
        
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # 给障碍物一个容易区分的颜色，比如灰色
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.compute_vertex_normals() # 计算法线，让光照效果更好
    print(f"成功加载障碍物: {mesh_path}")

    # 2. 加载采样点云
    points_path = os.path.join(SCENE_PATH, SAMPLED_POINTS_FILE)
    if not os.path.exists(points_path):
        print(f"错误: 找不到采样点文件: {points_path}")
        return
        
    # sampled_points 的形状是 [N, 6]，我们只需要前3列 (x0)
    sampled_points_x0 = np.load(points_path)[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points_x0)
    # 给采样点一个鲜艳的颜色，比如红色
    pcd.paint_uniform_color([1, 0, 0])
    print(f"成功加载 {len(sampled_points_x0)} 个采样点: {points_path}")

    # 3. 创建坐标系和可视化窗口
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    print("\n--- 正在启动可视化窗口... ---")
    print("您可以用鼠标拖动、旋转、缩放。按 'q' 键关闭窗口。")
    o3d.visualization.draw_geometries([mesh, pcd, coordinate_frame])


if __name__ == '__main__':
    visualize()