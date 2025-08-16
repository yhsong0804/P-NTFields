# ==============================================================================
# 最终版：格式转换脚本 convert_off_to_obj.py
#
# 核心功能:
# 1. 读取一个指定路径下的 .off 格式的三维模型文件。
# 2. 将其几何信息（顶点和面片）无损地、原封不动地另存为 .obj 格式。
# 3. 这个脚本是我们进行“可视化诊断”的关键工具，能帮助我们在Blender中
#    亲眼验证我们每一步预处理（如归一化、合并）的结果是否正确。
# ==============================================================================

import igl
import os

# --- 您需要修改的配置 ---
# 1. 设置包含 .off 文件的场景路径
#    例如: './datasets/gibson_3smalltree/my_final_scene/0/'
SCENE_PATH = './datasets/gibson_3smalltree/my_final_scene/0/'

# 2. 设置要转换的 .off 文件名
OFF_FILENAME = 'mesh_z_up_scaled.off'

# 3. 设置您希望输出的 .obj 文件名
OBJ_FILENAME = 'FINAL_SCENE_FOR_VISUALIZATION.obj'
# --- 配置结束 ---


def convert():
    """
    读取.off文件并将其另存为.obj文件。
    """
    input_path = os.path.join(SCENE_PATH, OFF_FILENAME)
    output_path = os.path.join(SCENE_PATH, OBJ_FILENAME)

    print(f"--- 正在读取: {input_path} ---")
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        print("请确认您已经成功运行了生成.off文件的相关预处理步骤。")
        return

    # 使用igl库读取.off文件的顶点(v)和面片(f)数据
    v, f = igl.read_triangle_mesh(input_path)
    print("文件读取成功！")

    # 使用igl库将顶点和面片数据，写入到一个新的.obj文件
    igl.write_triangle_mesh(output_path, v, f)
    
    print(f"--- 成功！已将场景转换为 .obj 格式并保存到: {output_path} ---")


if __name__ == '__main__':
    convert()