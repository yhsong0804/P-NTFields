# preprocess_dynamic.py
# 实际就完成了off（scaling）
import os
import glob
import igl
import numpy as np
import traceback

# --- 您需要修改的配置 ---
# 设置您的动态数据集场景路径
# 注意：这个路径应该指向包含 t0, t1... 等文件的那个场景文件夹
#SCENE_PATH = './datasets/gibson_dynamic/0/' 
SCENE_PATH = './datasets/gibson_dynamic/1/' 
# --- 配置结束 ---

def preprocess_dynamic_scene():
    print(f"--- 开始处理动态场景: {SCENE_PATH} ---")

    # 1. 找到所有时刻的原始obj文件
    obj_files = sorted(glob.glob(os.path.join(SCENE_PATH, 'mesh_t*.obj')))
    if not obj_files:
        print(f"错误: 在'{SCENE_PATH}'中找不到 'mesh_t*.obj' 文件。")
        return

    print("找到以下原始动态模型文件:")
    for f in obj_files:
        print(f" - {f}")

    # 2. 加载所有mesh，并计算一个统一的、包含所有轨迹的包围盒
    all_vertices = []
    all_meshes = []
    for obj_file in obj_files:
        try:
            v, f = igl.read_triangle_mesh(obj_file)
            all_vertices.append(v)
            all_meshes.append({'v': v, 'f': f, 'name': os.path.basename(obj_file)})
        except Exception as e:
            print(f"读取文件 {obj_file} 时出错: {e}")
            return
    
    # 将所有顶点合并成一个大的点云，来计算总的包围盒
    combined_vertices = np.vstack(all_vertices)
    
    # 计算统一的包围盒和缩放参数
    bb_max = combined_vertices.max(axis=0, keepdims=True)
    bb_min = combined_vertices.min(axis=0, keepdims=True)
    
    # 统一的中心点和缩放比例
    unified_center = (bb_max + bb_min) / 2.0
    unified_scale = (bb_max - bb_min).max() # 使用最大边长作为缩放基准

    print("\n--- 计算出统一的归一化参数 ---")
    print(f"统一中心点: {unified_center}")
    print(f"统一缩放比例 (基于最大轨迹边长): {unified_scale}")

    # 3. 使用【同一套】参数，对每一个时刻的mesh进行归一化并保存
    print("\n--- 开始对每个时刻的模型进行归一化 ---")
    for mesh_data in all_meshes:
        try:
            v_original = mesh_data['v']
            f_original = mesh_data['f']
            
            # 应用统一的变换
            v_normalized = (v_original - unified_center) / unified_scale
            
            # 定义输出文件名
            file_name_without_ext = os.path.splitext(mesh_data['name'])[0]
            output_file = os.path.join(SCENE_PATH, f"{file_name_without_ext}_scaled.off")

            # 写入归一化后的 .off 文件
            igl.write_triangle_mesh(output_file, v_normalized, f_original)
            print(f"成功保存: {output_file}")

        except Exception as e:
            print(f"处理并保存 {mesh_data['name']} 时出错: {e}")
            print(traceback.format_exc())
            
    print("\n--- 动态场景归一化完成！ ---")

if __name__ == '__main__':
    preprocess_dynamic_scene()