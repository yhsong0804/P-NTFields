import sys
sys.path.append('.')

import torch
import pytorch_kinematics as pk
import numpy as np

import matplotlib
import matplotlib.pylab as plt
import sys
sys.path.append('.')

import open3d as o3d

from models import model_res_sigmoid as md 
from timeit import default_timer as timer
import igl


def Arm_FK(sampled_points, out_path_, path_name_, end_effect_):
    shape = sampled_points.shape
    pointsize = 0

    # 强制使用 float32 而不是 float64
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32   #  <<<=== 【修改】从原先的 float64 改为 float32

    # 构建机械臂链并把它转到 float32、GPU 上
    chain = pk.build_serial_chain_from_urdf(
        open(out_path_ + '/' + path_name_ + ".urdf").read(), end_effect_)
    chain = chain.to(dtype=dtype, device=d)   #  <<<=== 【修改】确保 chain 是 float32

    scale = np.pi / 0.5

    # 原先写法（会生成 float64 张量）：
    # th_batch = torch.tensor(scale*sampled_points, requires_grad=True).cuda()

    # 【修改】把 sampled_points 转成 float32，然后再移到 GPU
    th_batch = torch.tensor(scale * sampled_points, dtype=torch.float32, device=d, requires_grad=True)  #  <<<=== 【修改】

    # 现在调用 forward_kinematics 就不会报“Double”类型错误
    tg_batch = chain.forward_kinematics(th_batch, end_only=False)

    p_list = []
    iter = 0
    pointsize = 0

    traj = []
    mesh_list = []

    for tg in tg_batch:
        if iter > 1:
            mesh = o3d.io.read_triangle_mesh(
                out_path_ + '/meshes/visual/' + tg.replace('_link','') + '.obj')
            mesh_list.append(mesh)
            v = np.asarray(mesh.vertices)

            nv = np.ones((v.shape[0], 4))
            pointsize = pointsize + v.shape[0]
            nv[:, :3] = v

            # 获取当前链接的变换矩阵 m（NumPy 数组），然后把它转成 torch.tensor
            m = tg_batch[tg].get_matrix()    # m 的 dtype 默认是 float64？
            # 【修改】把 m 转成 float32，再移到 GPU
            m = torch.tensor(m, dtype=torch.float32, device=d)   #  <<<=== 【修改】

            t = torch.from_numpy(nv).to(dtype=torch.float32, device=d)  #  <<<=== 【修改】
            p = torch.matmul(m, t.T)

            # p 的形状应该是 [4, N]，后面 permute 得到 [1, N, 4] 之类
            p = torch.permute(p, (0, 2, 1))
            p_list.append(p)

            # 释放局部变量
            del m, p, t, nv, v

        iter = iter + 1

    wholemesh = o3d.geometry.TriangleMesh()

    for ii in range(len(mesh_list)):
        mesh = mesh_list[ii]
        # p_list[ii] 已经是 float32 的 torch.Tensor
        p = p_list[ii].detach().cpu().numpy()
        for jj in range(len(p)):
            pp = p[jj]
            mesh.vertices = o3d.utility.Vector3dVector(pp[:, :3])
            wholemesh += mesh

    wholemesh.compute_vertex_normals()
    return wholemesh


modelPath = './Experiments/UR5'
dataPath  = './datasets/arm/UR5'

model = md.Model(modelPath, dataPath, 6, device='cuda')
model.load('./Experiments/UR5/Model_Epoch_10000_ValLoss_3.526511e-03.pt')


#for ii in range(10):

for ii in range(3):
    # 原先写法（会产生 float64）：
    # XP = torch.tensor([[0.00,0.0,0.0,-0.00,0.00,-0.00,
    #                     -1.3, 0.4, 1.1, 0.5,-0.5,0.0]]).cuda()
    # XP = torch.tensor([[-2.2, 0.4, 1.1, 0.5,-0.5,0.9,
    #                     -1.3, 0.4, 1.1, 0.5,-0.5,0.0]]).cuda()

    # 【修改】把列表数据明确指定为 float32，再移到 GPU
    XP = torch.tensor([[0.00, 0.0, 0.0, -0.00, 0.00, -0.00,
                        -1.3,  0.4,  1.1,  0.5, -0.5,  0.0]],
                      dtype=torch.float32, device='cuda')    #  <<<=== 【修改】

    XP = torch.tensor([[-2.2,  0.4,  1.1,  0.5, -0.5,  0.9,
                         -1.3,  0.4,  1.1,  0.5, -0.5,  0.0]],
                      dtype=torch.float32, device='cuda')    #  <<<=== 【修改**

    # BASE 中原先也是 float64，这里也要指定为 float32
    BASE = torch.tensor([[0, -0.5 * np.pi, 0.0, -0.5 * np.pi, 0.0, 0.0,
                          0, -0.5 * np.pi, 0.0, -0.5 * np.pi, 0.0, 0.0]],
                        dtype=torch.float32, device='cuda')  #  <<<=== 【修改】

    XP = XP + BASE

    scale = np.pi / 0.5
    # XP 现在是 float32，下面这样除以 scale 时保持 float32
    XP = XP / scale

    dis = torch.norm(XP[:, 6:] - XP[:, :6])

    point0 = []
    point1 = []
    point0.append(XP[:, :6])
    point1.append(XP[:, 6:])

    start = timer()
    iter = 0

    while dis > 0.03:
        # 计算梯度，model.Gradient 返回 float32 张量
        gradient = model.Gradient(XP.clone())

        XP = XP + 0.015 * gradient
        dis = torch.norm(XP[:, 6:] - XP[:, :6])

        point0.append(XP[:, :6])
        point1.append(XP[:, 6:])

        iter = iter + 1
        if iter > 300:
            break

    end1 = timer()
    print("plan", end1 - start)

    point1.reverse()
    point = point0 + point1

    # 在这里把所有轨迹拼成 numpy 数组 float32 再使用
    xyz = torch.cat(point).to('cpu').data.numpy().astype(np.float32)  #  <<<=== 【修改】

    # 只取第一个和最后一个配置作为起终点测试机械臂正运动学
    xyz0 = np.zeros((2, 6), dtype=np.float32)  #  <<<=== 【修改】
    xyz0[0, :] = xyz[0, :]
    xyz0[1, :] = xyz[-1, :]

    scale = np.pi / 0.5
    # 传给 Arm_FK 的 sampled_points 也要保持 float32
    wholemesh = Arm_FK(xyz[0::1, :], 'datasets/arm/UR5', 'UR5', 'wrist_3_link')

    def length(path):
        size = path.shape[0]
        l = 0
        for i in range(size - 1):
            l += np.linalg.norm(path[i + 1, :] - path[i, :])
        return l

    print(length(xyz))

    file_path = 'datasets/arm/'
    mesh_name = 'untitled_scaled.off'
    path = file_path + 'UR5' + '/'

    obstacle = o3d.io.read_triangle_mesh(path + mesh_name)
    vertices = np.asarray(obstacle.vertices)
    faces = np.asarray(obstacle.triangles)
    obstacle.vertices = o3d.utility.Vector3dVector(vertices)

    obstacle.compute_vertex_normals()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([obstacle, wholemesh, mesh_frame])
