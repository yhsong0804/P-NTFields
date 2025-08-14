# identify_components.py
import open3d as o3d
import numpy as np
import os

# --- 您需要修改的配置 ---
# 1. 场景路径
SCENE_PATH = './datasets/gibson_3smalltree/my_final_scene/0/' 

# 2. 障碍物模型文件名
OBSTACLE_FILE = 'mesh_z_up_scaled.off'

# 3. 设置一个用于存放导出组件的文件夹名
OUTPUT_FOLDER = 'components_for_blender'
# --- 配置结束 ---


def identify_and_export():
    """
    主函数，用于分离组件，打印信息，并导出为独立的 .obj 文件。
    """
    print("--- 开始加载障碍物模型... ---")
    mesh_path = os.path.join(SCENE_PATH, OBSTACLE_FILE)
    if not os.path.exists(mesh_path):
        print(f"错误: 找不到障碍物文件: {mesh_path}")
        return
    combined_mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("障碍物模型加载成功。")

    # 创建用于存放导出文件的文件夹
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出文件夹: ./{OUTPUT_FOLDER}/")

    print("\n--- 正在分离独立的障碍物组件... ---")
    triangle_clusters, cluster_n_triangles, _ = combined_mesh.cluster_connected_triangles()
    
    triangle_clusters = np.asarray(triangle_clusters)
    n_triangles = np.asarray(cluster_n_triangles)
    
    clusters_with_sizes = sorted(enumerate(n_triangles), key=lambda x: x[1], reverse=True)
    
    print(f"检测到 {len(clusters_with_sizes)} 个独立组件。正在导出前10大组件...")
    
    vertices = np.asarray(combined_mesh.vertices)
    triangles = np.asarray(combined_mesh.triangles)

    for i, (cluster_id, size) in enumerate(clusters_with_sizes[:10]):
        # 从主网格中提取出当前组件的三角面
        component_triangle_indices = np.where(triangle_clusters == cluster_id)[0]
        component_triangles = triangles[component_triangle_indices]
        
        # 创建一个新的、只包含当前组件的网格对象
        component_mesh = o3d.geometry.TriangleMesh()
        component_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        component_mesh.triangles = o3d.utility.Vector3iVector(component_triangles)
        
        # 清理网格，移除未被引用的顶点，以减小文件大小
        component_mesh.remove_unreferenced_vertices()

        # 定义输出文件名
        output_filename = os.path.join(OUTPUT_FOLDER, f"ID_{cluster_id}_size_{size}.obj")
        
        # 导出为 .obj 文件
        o3d.io.write_triangle_mesh(output_filename, component_mesh)
        print(f"  - 已导出: {output_filename}")

    print("\n" + "*"*20 + " 请执行下一步 " + "*"*20)
    print(f"请打开 Blender，然后导入 './{OUTPUT_FOLDER}/' 文件夹中的所有 .obj 文件。")
    print("通过在Blender中点选，找到小石头对应的那个文件，从文件名中记下它的ID。")
    print("最后，回到 quantify_and_identify.py 脚本，将 TARGET_CLUSTER_ID 修改为您找到的ID，再次运行即可得到最终量化结果。")


if __name__ == '__main__':
    identify_and_export()