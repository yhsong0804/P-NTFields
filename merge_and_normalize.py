# ==============================================================================
# 最终版：合并与归一化脚本 merge_and_normalize.py
#
# 核心功能:
# 1. 加载多个独立的.obj模型文件。
# 2. 将它们合并成一个单一的、包含了所有物体的场景。
# 3. 对整个合并后的场景，进行统一的、与原始代码完全等价的“非等比拉伸”归一化。
# 4. 将最终的、比例正确的完美场景，保存为单个.obj文件。
# ==============================================================================

import igl
import numpy as np
import os

# --- 您需要修改的配置 ---
# 1. 定义所有需要合并的原始模型文件的路径列表
#    第一个通常是主场景，后面是您要添加的物体
#    【请注意】这里的路径是您存放原始、未归一化.obj文件的地方
INPUT_FILES = [
    #'./datasets/gibson_3smalltree/0/original_gibson.obj', # <--- 修改这里：原始Gibson场景的路径
    #'./datasets/gibson_3smalltree/0/large_rock_1.obj',    # <--- 修改这里：大石头1的路径
    #'./datasets/gibson_3smalltree/0/large_rock_2.obj',    # <--- 修改这里：大石头2的路径
    #'./datasets/gibson_3smalltree/0/small_rock.obj'       # <--- 修改这里：小石头的路径
    './datasets/gibson_3smalltree/1/my_perfect_scene_2.5.obj', # <--- 修改这里：增大后直接合并场景的路径
]

# 2. 定义您希望输出的、合并并归一化后的【单个】场景文件的路径
#    建议在一个全新的数据集文件夹中创建它
OUTPUT_FILE = './datasets/gibson_3smalltree/my_final_scene/1/mesh_z_up.obj' # <--- 修改这里：最终输出路径
# --- 配置结束 ---


def merge_and_normalize():
    """
    加载多个.obj文件，将它们合并成一个单一的场景，
    然后对整个场景进行统一的归一化，最后保存为单个.obj文件。
    """
    all_vertices = []
    all_faces = []
    
    print("--- 开始加载并合并模型... ---")
    
    current_vertex_offset = 0
    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，已跳过。")
            continue
        
        # 使用igl库读取每个模型
        v, f = igl.read_triangle_mesh(file_path)
        
        # 将顶点数据直接追加到总列表
        all_vertices.append(v)
        
        # 【关键】合并面片时，需要加上当前的顶点偏移量
        # 这确保了每个模型的面片索引，能指向合并后列表中的正确顶点
        # 这个逻辑与Blender中的Ctrl+J操作是等价的
        all_faces.append(f + current_vertex_offset)
        
        # 更新下一个模型的顶点偏移量
        current_vertex_offset += v.shape[0]
        
        print(f"成功加载并处理: {os.path.basename(file_path)}")

    if not all_vertices:
        print("错误: 没有加载到任何有效的模型，程序退出。")
        return

    # 将列表中的所有顶点和面片数据，合并成一个巨大的Numpy数组
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)
    
    print(f"\n--- 合并完成！总顶点数: {len(combined_vertices)}, 总面片数: {len(combined_faces)} ---")

    # --- 对【整个合并后的场景】进行统一的归一化 (采用与原始代码相同的“非等比拉伸”策略) ---
    
    # 1. 计算【集体照】的总包围盒
    bb_max = combined_vertices.max(axis=0)
    bb_min = combined_vertices.min(axis=0)
    
    # 2. 计算【集体照】的中心点
    center = (bb_max + bb_min) / 2.0
    
    # 3. 【核心修正】: 不再使用单一的scale值，而是使用每个轴独立的缩放比例
    #    这行代码的逻辑，现在与原始的 convert_to_scaled_off.py 完全等价，
    #    它会强行将场景在每个轴向上都拉伸或压缩，以填满[-0.5, 0.5]的空间。
    normalized_vertices = (combined_vertices - center) / (bb_max - bb_min)
    
    print("--- 场景统一归一化完成 (采用非等比拉伸策略) ---")

    # --- 保存最终的、单一的场景文件 ---
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    igl.write_triangle_mesh(OUTPUT_FILE, normalized_vertices, combined_faces)
    
    print(f"\n--- 成功！已将合并并归一化后的最终场景保存到: {OUTPUT_FILE} ---")


if __name__ == '__main__':
    merge_and_normalize()