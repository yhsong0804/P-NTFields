# quantify_samples_final.py
# 考虑小石头是第几大的
import open3d as o3d
import numpy as np
import os
import torch
import bvh_distance_queries

# --- 您需要修改的配置 ---
# 1. 设置场景路径
SCENE_PATH = './datasets/gibson_3smalltree/my_final_scene/0/' 

# 2. 设置障碍物模型文件名
OBSTACLE_FILE = 'mesh_z_up_scaled.off'

# 3. 设置两组采样点文件名
BASELINE_POINTS_FILE = 'sampled_points_baseline.npy'
IMPROVED_POINTS_FILE = 'sampled_points_improved.npy' 

# 4. 设置距离阈值
SURFACE_THRESHOLD = 0.005
# --- 配置结束 ---


def quantify_results():
    """
    主函数，用于加载数据、分离小石头并进行定量分析。
    """
    print("--- 开始加载障碍物模型... ---")
    mesh_path = os.path.join(SCENE_PATH, OBSTACLE_FILE)
    if not os.path.exists(mesh_path):
        print(f"错误: 找不到障碍物文件: {mesh_path}")
        return
    
    combined_mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("障碍物模型加载成功。")

    # --- 1. 改进的小石头分离逻辑 ---
    print("\n--- 正在分离独立的障碍物组件... ---")
    triangle_clusters, cluster_n_triangles, _ = combined_mesh.cluster_connected_triangles()
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    # 按大小降序排列所有分离出的组件
    clusters_with_sizes = sorted(enumerate(cluster_n_triangles), key=lambda x: x[1], reverse=True)
    
    # 【新增】打印出前5大组件的大小，方便您验证
    print("检测到的前5大组件（按三角面数量排序）:")
    for i, (cluster_id, size) in enumerate(clusters_with_sizes[:5]):
        print(f"  - 第 {i+1} 大组件: ID={cluster_id}, 三角面数={size}")

    if len(clusters_with_sizes) < 4:
        print("错误：场景中的独立障碍物件数少于4，无法确定哪个是'小石头'。")
        return
    
    # 【已修改】根据您的分析，小石头是第四大的物体
    small_stone_cluster_id = clusters_with_sizes[4][0]
    
    small_stone_indices = np.where(triangle_clusters == small_stone_cluster_id)[0]
    small_stone_triangles = np.asarray(combined_mesh.triangles)[small_stone_indices]

    small_stone_mesh = o3d.geometry.TriangleMesh()
    small_stone_mesh.vertices = combined_mesh.vertices 
    small_stone_mesh.triangles = o3d.utility.Vector3iVector(small_stone_triangles)
    
    print(f"\n成功定位目标小石头！(ID={small_stone_cluster_id}) 它由 {len(small_stone_mesh.triangles)} 个三角面组成。")


    # --- 2. 定义使用 BVH 的统计函数 ---
    def count_points_on_surface(points_filepath, stone_mesh):
        """使用 bvh-distance-queries 计算并统计落在表面的点数"""
        points_path = os.path.join(SCENE_PATH, points_filepath)
        if not os.path.exists(points_path):
            print(f"警告: 找不到采样点文件: {points_path}")
            return 0, None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        stone_vertices_tensor = torch.tensor(np.asarray(stone_mesh.vertices), dtype=torch.float32, device=device)
        stone_faces_tensor = torch.tensor(np.asarray(stone_mesh.triangles), dtype=torch.long, device=device)
        stone_triangles_tensor = stone_vertices_tensor[stone_faces_tensor].unsqueeze(0)

        points_data = np.load(points_path)[:, :3]
        query_points_tensor = torch.tensor(points_data, dtype=torch.float32, device=device).unsqueeze(0)
        
        bvh = bvh_distance_queries.BVH()
        squared_distances, _, _, _ = bvh(stone_triangles_tensor, query_points_tensor)
        
        distances = torch.sqrt(squared_distances.squeeze()).cpu().numpy()
        
        on_surface_count = np.sum(distances < SURFACE_THRESHOLD)
        return on_surface_count, len(points_data)

    # --- 3. 执行统计并生成报告 ---
    print(f"\n--- 开始统计落在表面 (阈值 < {SURFACE_THRESHOLD}) 的点数... ---")
    
    baseline_count, baseline_total = count_points_on_surface(BASELINE_POINTS_FILE, small_stone_mesh)
    improved_count, improved_total = count_points_on_surface(IMPROVED_POINTS_FILE, small_stone_mesh)

    print("\n" + "="*40)
    print("           定量分析结果 (最终版)")
    print("="*40)
    if baseline_total:
        print(f"原始算法 (Baseline):")
        print(f"  - 落在小石头表面的点数: {baseline_count} / {baseline_total} ({baseline_count/baseline_total:.2%})")
    
    if improved_total:
        print(f"改进算法 (Improved):")
        print(f"  - 落在小石头表面的点数: {improved_count} / {improved_total} ({improved_count/improved_total:.2%})")
    
    if baseline_count > 0 and improved_count > 0:
        increase_percent = ((improved_count - baseline_count) / baseline_count) * 100
        print("\n--- 对比结论 ---")
        print(f"改进算法在小石头表面的采样点数比原始算法提升了: {increase_percent:.2f}%")
        
    print("="*40)


if __name__ == '__main__':
    quantify_results()