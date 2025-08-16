# verify_and_compare.py
import open3d as o3d
import numpy as np
import os

# --- 您需要修改的配置 ---
# 1. 设置您要检查的场景路径
#    确保这个路径下有障碍物模型和两份采样点文件
SCENE_PATH = './datasets/gibson_3smalltree/my_final_scene/0/' 

# 2. 设置要加载的障碍物模型文件名 (归一化后的 .off 文件)
OBSTACLE_FILE = 'mesh_z_up_scaled.off'

# 3. 设置两组采样点文件名
#    - BASELINE_POINTS_FILE: 原始算法生成的采样点
#    - IMPROVED_POINTS_FILE: 您的粗细两阶段采样算法生成的点
BASELINE_POINTS_FILE = 'sampled_points_baseline.npy'
IMPROVED_POINTS_FILE = 'sampled_points_improved.npy' 
# --- 配置结束 ---


def visualize_comparison():
    """
    可视化对比原始采样和改进采样的效果。
    """
    print("--- 开始加载数据进行对比可视化... ---")
    
    # 1. 加载障碍物模型 (大石头和小石头)
    mesh_path = os.path.join(SCENE_PATH, OBSTACLE_FILE)
    if not os.path.exists(mesh_path):
        print(f"错误: 找不到障碍物文件: {mesh_path}")
        return
        
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # 给障碍物一个中性的灰色，以便突出显示采样点
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.compute_vertex_normals()
    print(f"成功加载障碍物: {mesh_path}")

    geometries_to_draw = [mesh]

    # 2. 加载并处理原始算法的采样点 (Baseline)
    baseline_points_path = os.path.join(SCENE_PATH, BASELINE_POINTS_FILE)
    if os.path.exists(baseline_points_path):
        # sampled_points 的形状是 [N, 6]，我们只需要前3列 (x0) 作为点的位置
        baseline_points_x0 = np.load(baseline_points_path)[:, :3]
        
        pcd_baseline = o3d.geometry.PointCloud()
        pcd_baseline.points = o3d.utility.Vector3dVector(baseline_points_x0)
        # 给原始采样点设置为【红色】
        pcd_baseline.paint_uniform_color([1, 0, 0])
        geometries_to_draw.append(pcd_baseline)
        print(f"成功加载 {len(baseline_points_x0)} 个【红色】原始采样点 (Baseline)")
    else:
        print(f"警告: 找不到原始采样点文件: {baseline_points_path}")

    # 3. 加载并处理改进算法的采样点 (Improved)
    improved_points_path = os.path.join(SCENE_PATH, IMPROVED_POINTS_FILE)
    if os.path.exists(improved_points_path):
        improved_points_x0 = np.load(improved_points_path)[:, :3]
        
        pcd_improved = o3d.geometry.PointCloud()
        pcd_improved.points = o3d.utility.Vector3dVector(improved_points_x0)
        # 给改进采样点设置为【绿色】，以形成鲜明对比
        pcd_improved.paint_uniform_color([0, 1, 0])
        geometries_to_draw.append(pcd_improved)
        print(f"成功加载 {len(improved_points_x0)} 个【绿色】改进采样点 (Improved)")
    else:
        print(f"警告: 找不到改进采样点文件: {improved_points_path}")

    # 4. 创建坐标系并启动可视化窗口
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries_to_draw.append(coordinate_frame)
    
    print("\n--- 正在启动可视化窗口... ---")
    print("对比说明:")
    print("  - 【灰色】: 障碍物 (大石头和小石头)")
    print("  - 【红色】: 原始算法的采样点")
    print("  - 【绿色】: 您的改进算法的采样点")
    print("请检查【绿色】点是否更密集地分布在小石头周围。")
    print("按 'q' 键关闭窗口。")
    
    o3d.visualization.draw_geometries(geometries_to_draw)


if __name__ == '__main__':
    visualize_comparison()