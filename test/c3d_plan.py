import sys

sys.path.append('.')
from models import model_res_sigmoid_multi as md
import torch
import os 
import numpy as np
import matplotlib.pylab as plt
import torch
from torch import Tensor
from torch.autograd import Variable, grad

from timeit import default_timer as timer
import math
import igl
import open3d as o3d

#wrong path? no influnce

modelPath = './Experiments/c3d'

dataPath = './datasets/c3d/'
womodel    = md.Model(modelPath, dataPath, 3, 2, device='cuda')

#rhe same .pt as gibson environment
womodel.load('./Experiments/c3d/Model_Epoch_10000_ValLoss_1.221157e-01.pt')#
womodel.network.eval()
    
max_x = 0
max_y = 0
max_z = 0
#for gib_id in range(2):
#gib_id可改为1
#gib_id = 1

#gib_id = 0
c3d_id = 3
v, f = igl.read_triangle_mesh("datasets/c3d/"+str(c3d_id)+"/model_scaled.off")        
print(c3d_id)

vertices=v*20
faces=f

vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
faces = torch.tensor(faces, dtype=torch.long, device='cuda')
triangles = vertices[faces].unsqueeze(dim=0)

B = np.load("datasets/c3d/"+str(c3d_id)+"/B.npy")
B = Variable(Tensor(B)).to('cuda')

#核心路径生成循环
for ii in range(5):
    #设置起点、终点
    start_goal = np.array([[-6,-7,-6,2,7,-2.5]])
    
    #转成pytorch tensor 放在GPU
    XP=start_goal
    XP = Variable(Tensor(XP)).to('cuda')
    XP=XP/20.0

    dis=torch.norm(XP[:,3:6]-XP[:,0:3])

    #开始计时
    start = timer()

    point0=[]
    point1=[]

    point0.append(XP[:,0:3].clone())
    point1.append(XP[:,3:6].clone())
    #print(id)
    #用梯度上升走最佳路径

    iter=0
    while dis>0.06:#起点和终点距离大于0.06,继续沿梯度方向移动
        gradient = womodel.Gradient(XP.clone(), B)#计算梯度
   
        XP = XP + 0.03 * gradient
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
xyz= torch.cat(point).to('cpu').data.numpy()#np.asarray(point)
#放大到真实尺度
xyz=20*xyz
#路径点作pointcloud
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)
#再读一次mesh 并scale+compute normals
mesh = o3d.io.read_triangle_mesh("datasets/c3d/"+str(c3d_id)+"/model_scaled.off")
        
mesh.scale(20, center=(0,0,0))

mesh.compute_vertex_normals()
#mesh和pcd一起传入
o3d.visualization.draw_geometries([mesh,pcd])


