# 测试动态数据处理
import sys
sys.path.append('.')

from dataprocessing.speed_sampling_dynamic import sample_dynamic_speed_sequence
import os

def test_dynamic_processing():
    """测试动态数据处理功能"""
    
    # 测试路径
    test_scene_path = "./datasets/gibson_3smalltree_dynamic/my_final_scene/0"
    
    # 检查是否存在时序mesh文件
    required_files = [f"mesh_z_up_t{t}.obj" for t in range(4)]
    missing_files = []
    
    for file in required_files:
        full_path = os.path.join(test_scene_path, file)
        if not os.path.exists(full_path):
            missing_files.append(file)
    
    if missing_files:
        print("缺少以下时序mesh文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请先创建这些文件再运行动态处理。")
        
        # 创建示例文件结构说明
        print("\n预期的文件结构:")
        print("datasets/gibson_3smalltree_dynamic/")
        print("└── my_final_scene/")
        print("    ├── 0/")
        print("    │   ├── mesh_z_up_t0.obj  # 时刻0的场景")
        print("    │   ├── mesh_z_up_t1.obj  # 时刻1的场景(障碍物移动)")
        print("    │   ├── mesh_z_up_t2.obj  # 时刻2的场景")
        print("    │   └── mesh_z_up_t3.obj  # 时刻3的场景")
        print("    └── 1/")
        print("        ├── mesh_z_up_t0.obj")
        print("        ├── mesh_z_up_t1.obj") 
        print("        ├── mesh_z_up_t2.obj")
        print("        └── mesh_z_up_t3.obj")
        return False
    else:
        print("发现所有时序mesh文件，开始处理...")
        try:
            # 执行动态采样
            sample_dynamic_speed_sequence(test_scene_path, 10000, 3, 4)
            print("动态数据处理成功！")
            return True
        except Exception as e:
            print(f"动态数据处理失败: {e}")
            return False

if __name__ == "__main__":
    test_dynamic_processing()