# gpt/build_scene.py

from pathlib import Path
import trimesh

# 1. 动态定位项目根目录（假定本脚本在 <project_root>/gpt/ 下）
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. 构造模型文件路径
tree_path = BASE_DIR / 'obj_tree' / 'Tree1_1.obj'
rock_path = BASE_DIR / 'obj_stone' / 'FlatCluster_04_LOD1.obj'

# 3. 确认文件存在，否则报错
if not tree_path.is_file():
    raise FileNotFoundError(f"Tree OBJ not found: {tree_path}")
if not rock_path.is_file():
    raise FileNotFoundError(f"Rock OBJ not found: {rock_path}")

# 4. 加载网格
tree = trimesh.load(tree_path.as_posix(), force='mesh')
rock = trimesh.load(rock_path.as_posix(), force='mesh')

# 5. 把树“落地”对齐：全局将最低点移到 Z=0
tree_min_z = tree.bounds[0][2]
tree.apply_translation((0, 0, -tree_min_z))

# 6. 计算包围盒尺寸（此时 tree 已对齐底部，rock 保持原位）
tree_min, tree_max = tree.bounds
rock_min, rock_max = rock.bounds
tree_size = tree_max - tree_min
rock_size = rock_max - rock_min

# 7. 计算石头缩放因子（示例：让石头最长边是树最长边的 50%）
scale_factor = (tree_size.max() / rock_size.max()) * 0.5

print(f"Tree size (dx,dy,dz) = {tree_size}")
print(f"Rock size (dx,dy,dz) = {rock_size}")
print(f"Scaling rock by factor = {scale_factor:.3f}")

# 8. 定义各实例在 XY 平面上的位置
tree_positions = [
    (0.0,   0.0,  0.0),
    (40.0, 10.0,  0.0),
    (10.0, 40.0,  0.0),
]
rock_positions = [
    (25.0, -15.0, 0.0),
    (-25.0, -10.0, 0.0),
]

# 9. 生成并收集所有实例几何
geoms = []

# 树实例：直接平移
for pos in tree_positions:
    m = tree.copy()
    m.apply_translation(pos)
    geoms.append(m)

# 石头实例：先缩放，再“落地”对齐，再平移
original_rock_min_z = rock.bounds[0][2]
for pos in rock_positions:
    m = rock.copy()
    m.apply_scale(scale_factor)
    # 缩放后将底部移到
