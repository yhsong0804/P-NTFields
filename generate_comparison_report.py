# 生成柱状图 大/小/整体数量对比
import open3d as o3d
import numpy as np
import os
import torch
import bvh_distance_queries
import matplotlib.pyplot as plt

# --- 请确认这里的配置 ---
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

# 5. 【关键】要分析的目标ID
#    格式：'自定义名称': ID
TARGETS = {
    'Small Stone': 2,
    'Large Stone A': 1339,
    'Large Stone B': 1743
}
# --- 配置结束 ---


def count_points_on_surface(points_filepath, target_mesh):
    """使用 bvh-distance-queries 计算并统计落在表面的点数"""
    points_path = os.path.join(SCENE_PATH, points_filepath)
    if not os.path.exists(points_path):
        print(f"警告: 找不到文件: {points_path}")
        return 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stone_vertices = torch.tensor(np.asarray(target_mesh.vertices), dtype=torch.float32, device=device)
    stone_faces = torch.tensor(np.asarray(target_mesh.triangles), dtype=torch.long, device=device)
    stone_triangles = stone_vertices[stone_faces].unsqueeze(0)

    points_data = np.load(points_path)[:, :3]
    query_points = torch.tensor(points_data, dtype=torch.float32, device=device).unsqueeze(0)

    bvh = bvh_distance_queries.BVH()
    sq_distances, _, _, _ = bvh(stone_triangles, query_points)

    distances = torch.sqrt(sq_distances.squeeze()).cpu().numpy()

    on_surface_count = np.sum(distances < SURFACE_THRESHOLD)
    return on_surface_count


def generate_report():
    """主函数，执行所有分析并生成文本报告和图表。"""
    print("--- 开始加载障碍物模型... ---")
    mesh_path = os.path.join(SCENE_PATH, OBSTACLE_FILE)
    if not os.path.exists(mesh_path):
        print(f"错误: 找不到障碍物文件: {mesh_path}")
        return
    combined_mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("障碍物模型加载成功。")

    print("\n--- 正在分离独立的障碍物组件... ---")
    triangle_clusters, _, _ = combined_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    triangles = np.asarray(combined_mesh.triangles)
    vertices = np.asarray(combined_mesh.vertices)

    results = {'Baseline': {}, 'Improved': {}}
    all_target_meshes = []

    # 1. 分别统计每个目标的点数
    for name, target_id in TARGETS.items():
        print(f"--- 正在分析目标: {name} (ID={target_id}) ---")
        target_indices = np.where(triangle_clusters == target_id)[0]
        if len(target_indices) == 0:
            print(f"警告：找不到ID为 {target_id} 的组件，已跳过。")
            continue

        target_triangles = triangles[target_indices]
        target_mesh = o3d.geometry.TriangleMesh()
        target_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        target_mesh.triangles = o3d.utility.Vector3iVector(target_triangles)
        all_target_meshes.append(target_mesh)

        results['Baseline'][name] = count_points_on_surface(BASELINE_POINTS_FILE, target_mesh)
        results['Improved'][name] = count_points_on_surface(IMPROVED_POINTS_FILE, target_mesh)

    # 2. 合并所有目标网格，统计整体点数
    print("\n--- 正在分析所有目标的整体情况... ---")
    overall_mesh = o3d.geometry.TriangleMesh()
    for m in all_target_meshes:
        overall_mesh += m
    
    if overall_mesh.has_triangles():
        overall_mesh.remove_duplicated_vertices()
        overall_mesh.remove_duplicated_triangles()
        results['Baseline']['Overall'] = count_points_on_surface(BASELINE_POINTS_FILE, overall_mesh)
        results['Improved']['Overall'] = count_points_on_surface(IMPROVED_POINTS_FILE, overall_mesh)
    else:
        print("警告：没有有效的目标网格可用于'Overall'统计。")
        results['Baseline']['Overall'] = 0
        results['Improved']['Overall'] = 0


    # 3. 打印文本报告
    print_text_report(results)

    # 4. 生成并保存图表
    create_bar_chart(results)


def print_text_report(results):
    """打印详尽的文本报告。"""
    print("\n" + "="*50)
    print(" " * 15 + "最终量化分析报告")
    print("="*50)
    print(f"{'Target':<15} | {'Baseline':>10} | {'Improved':>10} | {'Change (%)':>12}")
    print("-"*50)

    for name in results['Baseline']:
        b_count = results['Baseline'][name]
        i_count = results['Improved'][name]
        if b_count > 0:
            change = ((i_count - b_count) / b_count) * 100
            change_str = f"{change:+.2f}%"
        else:
            change_str = "N/A"
        print(f"{name:<15} | {b_count:>10} | {i_count:>10} | {change_str:>12}")
    print("="*50)


def create_bar_chart(results):
    """生成并保存柱状对比图。"""
    labels = list(results['Baseline'].keys())
    baseline_counts = list(results['Baseline'].values())
    improved_counts = list(results['Improved'].values())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, baseline_counts, width, label='Baseline', color='crimson', alpha=0.8)
    rects2 = ax.bar(x + width/2, improved_counts, width, label='Improved', color='forestgreen', alpha=0.8)

    ax.set_ylabel('Number of Points on Surface')
    ax.set_title('Comparison of Sampling Effectiveness by Target')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱状图上显示具体数值
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    
    output_filename = 'comparison_chart.png'
    plt.savefig(output_filename)
    print(f"\n图表已成功保存为: {output_filename}")


if __name__ == '__main__':
    generate_report()