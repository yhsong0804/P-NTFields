#预处理-网络缩放的脚步
import igl
import numpy as np

# 原文件和缩放后文件
orig = 'datasets/gibson/0/mesh_z_up.obj'
scaled = 'datasets/gibson/0/mesh_z_up_scaled.off'

# 读两者
v0, f0 = igl.read_triangle_mesh(orig)
v1, f1 = igl.read_triangle_mesh(scaled)

# 计算包围盒
min0, max0 = v0.min(axis=0), v0.max(axis=0)
min1, max1 = v1.min(axis=0), v1.max(axis=0)

print('原始 OBJ 包围盒:')
print('  min:', min0, ' max:', max0)
print('缩放后 OFF 包围盒:')
print('  min:', min1, ' max:', max1)
print('边长范围:', (max1-min1))
