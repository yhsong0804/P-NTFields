import sys #sys系统模块

sys.path.append('.')
from models import model_res_sigmoid_multi as md
import torch
import os #os操作系统
import numpy as np
import matplotlib.pylab as plt
import torch
from torch import Tensor
from torch.autograd import Variable, grad

from timeit import default_timer as timer
import math
import igl#libigl处理网络（三角面、点云）
import open3d as o3d

#wrong path? no influnce
#modelPath = './Experiments/Gib'
modelPath = './Experiments/Gib_multi'

dataPath = './datasets/gibson/'
womodel    = md.Model(modelPath, dataPath, 3, 2, device='cuda')


womodel.load('./Experiments/Gib_multi/Model_Epoch_10000_ValLoss_1.221157e-01.pt')#加载训练好的模型参数
womodel.network.eval() #模型设为测试模式
    
max_x = 0
max_y = 0
max_z = 0
#for gib_id in range(2):
#gib_id可改为1
#gib_id = 1

gib_id = 0
v, f = igl.read_triangle_mesh("datasets/gibson/"+str(gib_id)+"/mesh_z_up_scaled.off")        
print(gib_id)

vertices=v*20#mesh坐标放大20倍
faces=f
#转化为pytorch张量，放到GPU上
vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
faces = torch.tensor(faces, dtype=torch.long, device='cuda')
triangles = vertices[faces].unsqueeze(dim=0)#获取三角面顶点坐标 增加一个batch维度 即[1, num_faces, 3, 3] 对应[1，面数，顶点数（三角形面片顶点数为3），xyz坐标]
#读取B.npy
B = np.load("datasets/gibson/"+str(gib_id)+"/B.npy")
B = Variable(Tensor(B)).to('cuda')

#核心路径生成循环
for ii in range(5):
    #设置起点、终点
    start_goal = np.array([[-6,-7,-6,2,7,-2.5]])
    
    #转成pytorch张量tensor 放在GPU
    XP=start_goal
    XP = Variable(Tensor(XP)).to('cuda')
    XP=XP/20.0
    #计算起点和终点距离
    dis=torch.norm(XP[:,3:6]-XP[:,0:3])

    #开始计时
    start = timer()

    point0=[]
    point1=[]
    #保存轨迹点的列表（point0记录起点轨迹）
    point0.append(XP[:,0:3].clone())#clone（）用于安全地保存当前状态
    point1.append(XP[:,3:6].clone())
    #print(id)
    #用梯度上升走最佳路径

    iter=0
    while dis>0.06:#起点和终点距离大于0.06,继续沿梯度方向移动（没汇合就继续）
        gradient = womodel.Gradient(XP.clone(), B)#计算梯度
   
        XP = XP + 0.03 * gradient#向梯度方向走，步长0.03
        dis=torch.norm(XP[:,3:6]-XP[:,0:3])
        #print(XP)
        #把新位置存到去程/回程列表（并非两条路径 指起点和终点同时往中间跑）
        point0.append(XP[:,0:3].clone())
        point1.append(XP[:,3:6].clone())
        iter=iter+1
        #防止无限循环
        if(iter>500):
            break
    #point0.append(p[:,3:6][0])

    end = timer()
    print("plan",end - start)
#去程倒序 再拼接
point1.reverse()
point=point0+point1
#点列表拼成（N，3）的numpy数组 
xyz= torch.cat(point).to('cpu').data.numpy()#np.asarray(point)合并为一个大张量 再转化为CPU的numpy数组
#放大到真实尺度
xyz=20*xyz
#路径点构造点云pointcloud，方便和mesh一起可视化
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)
#再读一次mesh 并scale+compute normals
mesh = o3d.io.read_triangle_mesh("datasets/gibson/"+str(gib_id)+"/mesh_z_up_scaled.off")
        
mesh.scale(20, center=(0,0,0))

mesh.compute_vertex_normals()
#mesh和pcd一起传入
o3d.visualization.draw_geometries([mesh,pcd])


