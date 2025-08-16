# gpt/wrap_scene.py

import trimesh
from pathlib import Path
import trimesh.transformations as tf

# 1. 直接指定容器内的 OBJ 路径
scene_path = Path('/workspace/P-NTFields/obj_received/3tree2stoneobj.obj')
out_path   = Path('/workspace/P-NTFields/wrapped_scene.obj')

# 2. 加载你的场景
scene = trimesh.load(str(scene_path), force='mesh')

# 3. 计算 AABB
min_corner, max_corner = scene.bounds
size = max_corner - min_corner

# 4. 加点余量并构造盒子
margin = size * 0.05
box_size = size + margin
center = (min_corner + max_corner) / 2.0

box_mesh = trimesh.creation.box(
    extents=box_size,
    transform=tf.translation_matrix(center)
)

# 5. 合并并导出
wrapped = trimesh.util.concatenate([scene, box_mesh])
wrapped.export(str(out_path))
print(f"Exported wrapped scene → {out_path}")
