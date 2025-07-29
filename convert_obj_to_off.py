# convert_obj_to_off.py
import igl
import os

# --- 您需要修改的配置 ---
# 1. 设置您合并后的.obj文件所在的场景路径
SCENE_PATH = './datasets/gibson_3smalltree/my_final_scene/0/'

# 2. 设置输入的.obj文件名
INPUT_OBJ = 'mesh_z_up.obj'

# 3. 【关键】设置输出的.off文件名，这个名字必须和下游脚本期望的一致
OUTPUT_OFF = 'mesh_z_up_scaled.off'
# --- 配置结束 ---


def convert():
    """
    读取.obj文件，并将其另存为下游脚本期望的_scaled.off文件。
    """
    input_path = os.path.join(SCENE_PATH, INPUT_OBJ)
    output_path = os.path.join(SCENE_PATH, OUTPUT_OFF)

    print(f"--- 正在读取: {input_path} ---")
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        print("请确认您已经成功运行了 merge_and_normalize.py 脚本。")
        return

    # 使用igl库读取.obj文件
    v, f = igl.read_triangle_mesh(input_path)
    print("文件读取成功！")

    # 使用igl库写入.off文件
    igl.write_triangle_mesh(output_path, v, f)
    print(f"--- 成功！已将场景转换为 .off 格式并保存到: {output_path} ---")


if __name__ == '__main__':
    convert()