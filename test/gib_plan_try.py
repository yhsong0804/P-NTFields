import sys
sys.path.append('.')

import numpy as np
import torch
from torch.autograd import Variable
from torch import Tensor
from timeit import default_timer as timer

import igl
import open3d as o3d

from models import model_res_sigmoid_multi as md

# ——— 加载模型 ———
modelPath = './Experiments/Gib'
dataPath  = './datasets/gibson/'
womodel   = md.Model(modelPath, dataPath, 3, 2, device='cuda')
womodel.load('./Experiments/Gib_multi/Model_Epoch_10000_ValLoss_1.221157e-01.pt')
womodel.network.eval()

# ——— 读取场景网格（用于碰撞检测） ———
gib_id = 1
v_np, f_np = igl.read_triangle_mesh(f"datasets/gibson/{gib_id}/mesh_z_up_scaled.off")

# ——— 读取随机映射矩阵 B ———
B = np.load(f"datasets/gibson/{gib_id}/B.npy")
B = Variable(Tensor(B)).to('cuda')

# ——— 设置起点和终点 ———
# 可以改成多个不同的起终点；这里演示一对
start_goal = np.array([[-6, -7, -6,   2,  7,  -2.5]])
XP = Variable(Tensor(start_goal)).to('cuda') / 20.0

# ——— 分别记录“去程”和“回程”轨迹 ———
point0 = [XP[:,0:3].clone()]  # 从起点开始
point1 = [XP[:,3:6].clone()]  # 从终点开始

# ——— 迭代沿梯度走向中点 ———
step_size = 0.01
dis = torch.norm(XP[:,3:6] - XP[:,0:3])
iter_count = 0
start = timer()

while dis > 0.06 and iter_count < 2000:
    # 1) 计算时间场梯度 ∇τ(XP)
    gradτ = womodel.Gradient(XP.clone(), B)
    # 2) 同步更新起点和终点
    XP = XP + step_size * gradτ
    dis = torch.norm(XP[:,3:6] - XP[:,0:3])

    # 3) 记录新位置
    point0.append(XP[:,0:3].clone())
    point1.append(XP[:,3:6].clone())

    iter_count += 1

end = timer()
print(f"Planned in {end-start:.2f}s over {iter_count} steps")

# ——— 合并去程+回程为完整轨迹 ———
point1.reverse()           # 先把回程轨迹倒序
full_path = point0 + point1

# ——— 转为 NumPy，并放大到真实尺度 ———
xyz = torch.cat(full_path).detach().cpu().numpy() * 20.0
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# ——— 载入原始网格并放大 ———
mesh = o3d.io.read_triangle_mesh(f"datasets/gibson/{gib_id}/mesh_z_up_scaled.off")
mesh.compute_vertex_normals()
mesh.scale(20.0, center=(0,0,0))

# ——— 弹窗显示 mesh + 轨迹点云 ———
o3d.visualization.draw_geometries([mesh, pcd])
