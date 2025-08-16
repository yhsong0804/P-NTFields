
#预处理——网格缩放——对比图
import open3d as o3d

# 加载原始
mesh0 = o3d.io.read_triangle_mesh('datasets/gibson/0/mesh_z_up.obj')
mesh0.compute_vertex_normals()
mesh0.paint_uniform_color([1,0,0])  # 红色

# 加载缩放后
mesh1 = o3d.io.read_triangle_mesh('datasets/gibson/0/mesh_z_up_scaled.off')
mesh1.compute_vertex_normals()
mesh1.paint_uniform_color([0,1,0])  # 绿色

# 把 mesh0 移一点，方便并排对比
mesh0.translate((-1.5, 0, 0))  

o3d.visualization.draw_geometries([mesh0, mesh1], window_name='原 vs 缩放后')
