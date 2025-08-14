# quantify_and_identify.py
# 继续找小石头
import open3d as o3d
import numpy as np
import os
import torch
import bvh_distance_queries

# --- 您需要修改的配置 ---
# 1. 场景路径
SCENE_PATH = './datasets/gibson_3smalltree/my_final_scene/0/' 
# 2. 障碍物模型文件名
OBSTACLE_FILE = 'mesh_z_up_scaled.off'
# 3. 采样点文件名
BASELINE_POINTS_FILE = 'sampled_points_baseline.npy'
#IMPROVED_POINTS_FILE = 'sampled_points_improved.npy' 
IMPROVED_POINTS_FILE = 'sampled_points.npy' 
# 4. 距离阈值  
SURFACE_THRESHOLD = 0.005

# --- 【第二步修改这里】 ---
# 在第一次运行后，将 None 替换为您确定的小石头的ID
TARGET_CLUSTER_ID = 2
# -------------------------


def main():
    """主函数，根据 TARGET_CLUSTER_ID 的设置，执行识别或量化。"""
    
    print("--- 开始加载障碍物模型... ---")
    mesh_path = os.path.join(SCENE_PATH, OBSTACLE_FILE)
    if not os.path.exists(mesh_path):
        print(f"错误: 找不到障碍物文件: {mesh_path}")
        return
    combined_mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("障碍物模型加载成功。")

    print("\n--- 正在分离独立的障碍物组件... ---")
    triangle_clusters, cluster_n_triangles, _ = combined_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    
    # 如果是“识别模式”
    if TARGET_CLUSTER_ID is None:
        print_cluster_info(combined_mesh, triangle_clusters, np.asarray(cluster_n_triangles))
        return
    
    # 如果是“量化模式”
    quantify_for_target(combined_mesh, triangle_clusters, TARGET_CLUSTER_ID)


def print_cluster_info(mesh, clusters, n_triangles):
    """【识别模式】打印所有组件的信息，帮助用户找到目标ID。"""
    clusters_with_sizes = sorted(enumerate(n_triangles), key=lambda x: x[1], reverse=True)
    
    print(f"检测到 {len(clusters_with_sizes)} 个独立组件。以下是前10大组件的信息:")
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    for i, (cluster_id, size) in enumerate(clusters_with_sizes[:10]):
        # 计算该组件的中心坐标
        cluster_triangle_indices = np.where(clusters == cluster_id)[0]
        cluster_vertex_indices = np.unique(triangles[cluster_triangle_indices].flatten())
        centroid = vertices[cluster_vertex_indices].mean(axis=0)
        
        print(f"  - 组件 {i+1} (ID={cluster_id}): 三角面数={size}, 中心坐标=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")

    print("\n" + "*"*20 + " 请执行第二步 " + "*"*20)
    print("请根据上面列出的中心坐标，结合您的场景布局，找出哪一个ID对应您的小石头。")
    print(f"然后修改本脚本第23行的 TARGET_CLUSTER_ID，并再次运行。")


def quantify_for_target(mesh, clusters, target_id):
    """【量化模式】针对指定ID的组件进行分析。"""
    
    # 从主网格中提取出目标组件
    target_indices = np.where(clusters == target_id)[0]
    if len(target_indices) == 0:
        print(f"错误：找不到ID为 {target_id} 的组件！请检查ID是否正确。")
        return
        
    target_triangles = np.asarray(mesh.triangles)[target_indices]
    target_mesh = o3d.geometry.TriangleMesh()
    target_mesh.vertices = mesh.vertices 
    target_mesh.triangles = o3d.utility.Vector3iVector(target_triangles)
    
    print(f"\n成功定位目标组件！(ID={target_id}) 它由 {len(target_mesh.triangles)} 个三角面组成。")
    
    print(f"\n--- 开始统计落在表面 (阈值 < {SURFACE_THRESHOLD}) 的点数... ---")
    baseline_count, baseline_total = count_points_on_surface(BASELINE_POINTS_FILE, target_mesh)
    improved_count, improved_total = count_points_on_surface(IMPROVED_POINTS_FILE, target_mesh)

    print_report(baseline_count, baseline_total, improved_count, improved_total)


def count_points_on_surface(points_filepath, stone_mesh):
    """使用 bvh-distance-queries 计算并统计落在表面的点数"""
    points_path = os.path.join(SCENE_PATH, points_filepath)
    if not os.path.exists(points_path): return 0, None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stone_vertices = torch.tensor(np.asarray(stone_mesh.vertices), dtype=torch.float32, device=device)
    stone_faces = torch.tensor(np.asarray(stone_mesh.triangles), dtype=torch.long, device=device)
    stone_triangles = stone_vertices[stone_faces].unsqueeze(0)

    points_data = np.load(points_path)[:, :3]
    query_points = torch.tensor(points_data, dtype=torch.float32, device=device).unsqueeze(0)
    
    bvh = bvh_distance_queries.BVH()
    sq_distances, _, _, _ = bvh(stone_triangles, query_points)
    
    distances = torch.sqrt(sq_distances.squeeze()).cpu().numpy()
    
    on_surface_count = np.sum(distances < SURFACE_THRESHOLD)
    return on_surface_count, len(points_data)


def print_report(b_count, b_total, i_count, i_total):
    """打印最终的量化报告。"""
    print("\n" + "="*40)
    print("           定量分析结果 (ID 定位版)")
    print("="*40)
    if b_total:
        print(f"原始算法 (Baseline):")
        print(f"  - 落在目标表面的点数: {b_count} / {b_total} ({b_count/b_total:.2%})")
    if i_total:
        print(f"改进算法 (Improved):")
        print(f"  - 落在目标表面的点数: {i_count} / {i_total} ({i_count/i_total:.2%})")
    if b_count > 0 and i_count > 0:
        increase = ((i_count - b_count) / b_count) * 100
        print("\n--- 对比结论 ---")
        print(f"改进算法在目标表面的采样点数比原始算法提升了: {increase:.2f}%")
    print("="*40)


if __name__ == '__main__':
    main()