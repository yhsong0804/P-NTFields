# ==============================================================================
# gemini_最终版本：静态场景“粗-细”两阶段采样脚本 speed_sampling_gpu.py
# 核心功能：
# 1. 保留了原始代码处理机械臂（arm）场景的所有功能。
# 2. 针对普通3D障碍物场景（如Gibson, c3d, test），将采样方法升级为
#    “粗-细”两阶段策略，以精确捕捉小尺寸障碍物（您的小石头）。
# ==============================================================================

import os 
import glob
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch
import pytorch_kinematics as pk

import bvh_distance_queries
import math
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Section 1: 机械臂（arm）专用函数 (来自原始代码，保持不变)
# ------------------------------------------------------------------------------

def bbox_corner(bbox):
    corner = torch.ones((4,8)).cuda()
    iter=0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corner[0,iter]=bbox[0+i*3]
                corner[1,iter]=bbox[1+j*3]
                corner[2,iter]=bbox[2+k*3]
                iter+=1
    return corner

def bbox_axis(bbox):
    axis = torch.zeros((3,3)).cuda()
    axis[0,0]=1
    axis[1,1]=1
    axis[2,2]=1
    return axis

def cross_matrix(transform_row):
    matrix = torch.zeros((transform_row.shape[0],3,3)).cuda()
    a = transform_row[:,0]
    b = transform_row[:,1]
    c = transform_row[:,2]
    l0 = torch.clamp(torch.sqrt(c**2+b**2), min=1e-6)
    l1 = torch.clamp(torch.sqrt(c**2+a**2), min=1e-6)
    l2 = torch.clamp(torch.sqrt(b**2+a**2), min=1e-6)
    matrix[:,0,1] = c/l0
    matrix[:,0,2] = -b/l0
    matrix[:,1,0] = -c/l1
    matrix[:,1,2] = a/l1
    matrix[:,2,0] = b/l2
    matrix[:,2,1] = -a/l2
    return matrix

def separate_axis(transform):
    axis = torch.zeros((transform.shape[0],15,3)).cuda()
    axis[:,0:3,:] = torch.eye(3).cuda()
    axis[:,3:6,:] = transform[:,0:3,0:3]
    axis[:,6:9,:]   = cross_matrix(transform[:,0,0:3])
    axis[:,9:12,:]  = cross_matrix(transform[:,1,0:3])
    axis[:,12:15,:] = cross_matrix(transform[:,2,0:3])
    return axis

def arm_obstacle_obb(th_batch, chain, out_path_, margin):
    rob_corner_list = []
    input_file_list = ['upper_arm','forearm','wrist_1','wrist_2','wrist_3']
    for input_file in input_file_list:
        bbox = np.load('./datasets/arm/UR5/meshes/collision/'+input_file+'bbox.npy')
        bbox = torch.tensor(bbox).cuda().float()
        bbox[:3] = bbox[:3] + margin
        bbox[3:] = bbox[3:] - margin
        rob_corner_list.append(bbox_corner(bbox))
    bbox = torch.tensor([0.7,-0.5,1.0,-0.6,-0.7,-0.7]).cuda().float()
    obs_points = bbox_corner(bbox)
    batch_size = 50000
    where_list = []
    for batch_id in range(math.floor(th_batch.shape[0]/batch_size)+1):
        if batch_id*batch_size==th_batch.shape[0]: break
        local_th_batch = th_batch[batch_id*batch_size:min((batch_id+1)*batch_size,th_batch.shape[0]),:]
        tg_batch = chain.forward_kinematics(local_th_batch, end_only = False)
        where_coll = torch.zeros(local_th_batch.shape[0], dtype=torch.bool).cuda()
        iter = 0
        for tg in tg_batch:
            if iter>2:
                m = tg_batch[tg].get_matrix()
                rob_points= m@rob_corner_list[iter-3]
                axis = separate_axis(m)
                rob_axis_point = axis@rob_points[:,0:3,:]
                obs_axis_point = axis@obs_points[0:3,:]
                rob_axis_point_min,_ = torch.min(rob_axis_point,dim=2)
                rob_axis_point_max,_ = torch.max(rob_axis_point,dim=2)
                obs_axis_point_min,_ = torch.min(obs_axis_point,dim=2)
                obs_axis_point_max,_ = torch.max(obs_axis_point,dim=2)
                where0 = rob_axis_point_max<obs_axis_point_min
                where1 = obs_axis_point_max<rob_axis_point_min
                where = torch.cat((where0,where1),dim=1)
                nonsep = torch.all((where == False),dim=1)
                combine = torch.cat((nonsep.unsqueeze(1),where_coll.unsqueeze(1)),dim=1)
                where_coll = torch.any(combine == True,dim=1)
                del m
            iter = iter+1
        where_list.append(nonsep)
    return torch.cat(where_list,0)

def arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs):
    whole_dis = []
    batch_size = 50000
    for batch_id in range(math.floor(th_batch.shape[0]/batch_size)+1):
        if batch_id*batch_size==th_batch.shape[0]: break
        tg_batch = chain.forward_kinematics(
            th_batch[batch_id*batch_size:min((batch_id+1)*batch_size,th_batch.shape[0]),:], end_only = False)
        p_list, p_size_list = [], []
        iter = 0
        for tg in tg_batch:
            if iter>1:
                v, f = igl.read_triangle_mesh(out_path_+'/meshes/collision/'+tg+'.obj')
                nv = np.ones((v.shape[0],4))
                p_size_list.append(v.shape[0])
                nv[:,:3]=v
                m = tg_batch[tg].get_matrix()
                t=torch.from_numpy(nv).float().cuda()
                p=torch.matmul(m[:],t.T)
                p = torch.permute(p, (0, 2, 1)).contiguous()
                p_list.append(p)
                del m,p,t,nv, v
            iter=iter+1
        pointsize = sum(p_size_list)
        p = torch.cat(p_list, dim=1)
        p = torch.reshape(p,(p.shape[0]*p.shape[1],p.shape[2])).contiguous()
        query_points = p[:,0:3].contiguous().unsqueeze(dim=0)
        bvh = bvh_distance_queries.BVH()
        distance, _, _, _= bvh(triangles_obs, query_points)
        distance = torch.sqrt(distance).squeeze()
        distance = torch.reshape(distance, (-1, pointsize))
        whole_distance,_ = torch.min(distance, dim=1)
        whole_dis.append(whole_distance)
        del p, p_list, tg_batch, distance, query_points, bvh,whole_distance
    return torch.cat(whole_dis, dim=0)

def arm_append_list(X_list, Y_list, chain, out_path_, triangles_obs, numsamples, dim, offset, margin):
    OutsideSize, WholeSize = numsamples + 2, 0
    scale = math.pi/0.5
    while OutsideSize > 0:
        P  = torch.rand((5*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((5*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((5*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(3)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        x0, x1 = P[PointsInside,:], nP[PointsInside,:]
        if(x0.shape[0]<=1): continue
        th_batch = scale*x0
        where = arm_obstacle_obb(th_batch, chain, out_path_, margin)
        x0, x1 = x0[where], x1[where]
        th_batch = scale*x0
        obs_distance0 = arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs)
        where_d = (obs_distance0 >= offset) & (obs_distance0 <= margin)
        x0, x1, y0 = x0[where_d], x1[where_d], obs_distance0[where_d]
        th_batch = scale*x1
        y1 = arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs)
        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)
        X_list.append(x); Y_list.append(y)
        OutsideSize -= x.shape[0]; WholeSize += x.shape[0]
        if(WholeSize > numsamples): break
    return X_list, Y_list

def arm_rand_sample_bound_points(numsamples, dim, v_obs, f_obs, offset, margin, out_path_ , path_name_, end_effect_):
    numsamples = int(numsamples)
    d = "cuda" if torch.cuda.is_available() else "cpu"
    chain = pk.build_serial_chain_from_urdf(open(out_path_+'/'+path_name_+".urdf").read(), end_effect_)
    chain = chain.to(dtype=torch.float32, device=d)
    t_obs = torch.tensor(v_obs[f_obs], dtype=torch.float32, device='cuda').unsqueeze(dim=0)
    X_list, Y_list = [], []
    X_list, Y_list = arm_append_list(X_list, Y_list, chain, out_path_, t_obs, numsamples, dim, offset, margin)
    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]    
    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance[:,0] , a_min = offset, a_max = margin)/margin
    speed[:,1] = np.clip(distance[:,1] , a_min = offset, a_max = margin)/margin
    return sampled_points, speed

# ------------------------------------------------------------------------------
# Section 2: 静态场景采样核心函数 (为解决小石头问题而重构)
# ------------------------------------------------------------------------------

def point_obstacle_distance(query_points, triangles_obs):
    """【核心工具函数】使用BVH在GPU上并行计算大量查询点到障碍物表面的最短距离。"""
    query_points = query_points.unsqueeze(dim=0)
    bvh = bvh_distance_queries.BVH()
    distances, _, _, _= bvh(triangles_obs, query_points)
    return torch.sqrt(distances).squeeze()

def point_append_list_static(X_list, Y_list, fine_sample_centers, triangles_obs, numsamples, dim, offset, margin):
    """【静态场景采样循环】在一个循环中不断生成和筛选点，直到满足数量要求。"""
    is_fine_sampling = fine_sample_centers is not None
    OutsideSize, WholeSize = numsamples + 2, 0
    if is_fine_sampling and fine_sample_centers.shape[0] == 0:
        return X_list, Y_list

    while OutsideSize > 0:
        batch_size = 8 * numsamples
        if is_fine_sampling:
            # --- 细采样逻辑：在敏感点中心周围生成新点 ---
            num_centers = fine_sample_centers.shape[0]
            center_indices = torch.randint(0, num_centers, (batch_size,), device='cuda')
            centers = fine_sample_centers[center_indices]
            perturbation = (torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5) * (2 * margin)
            P = centers + perturbation
        else:
            # --- 粗采样逻辑：在整个空间内均匀生成随机点 ---
            P  = torch.rand((batch_size,dim),dtype=torch.float32, device='cuda')-0.5
        
        dP = torch.rand((batch_size,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((batch_size,1),dtype=torch.float32, device='cuda'))*np.sqrt(dim)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL

        PointsInside = torch.all((nP <= 0.5) & (nP >= -0.5), dim=1) & torch.all((P <= 0.5) & (P >= -0.5), dim=1)
        x0, x1 = P[PointsInside,:], nP[PointsInside,:]
        
        if x0.shape[0] <= 1: continue

        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        where_d = (obs_distance0 > offset) & (obs_distance0 < margin)
        
        if where_d.sum() == 0: continue
            
        x0_f, x1_f, y0_f = x0[where_d], x1[where_d], obs_distance0[where_d]
        y1_f = point_obstacle_distance(x1_f, triangles_obs)
        
        X_list.append(torch.cat((x0_f, x1_f), 1))
        Y_list.append(torch.cat((y0_f.unsqueeze(1), y1_f.unsqueeze(1)), 1))
        
        OutsideSize -= x0_f.shape[0]; WholeSize += x0_f.shape[0]
        stage = "Fine" if is_fine_sampling else "Coarse"
        print(f"Static {stage} Sampling: collected {WholeSize}/{numsamples} points...")
        if WholeSize >= numsamples: break
        
    return X_list, Y_list

def point_rand_sample_bound_points(numsamples, dim, v, f, offset, margin):
    """【静态场景总调度】执行完整的“粗-细”两阶段采样流程。"""
    numsamples = int(numsamples)
    triangles_obs = torch.tensor(v[f], dtype=torch.float32, device='cuda').unsqueeze(0)
    
    # 1. 粗采样阶段
    print("--- Running Coarse Sampling Stage ---")
    coarse_num = int(numsamples * 0.7)
    X_list, Y_list = point_append_list_static([], [], None, triangles_obs, coarse_num, dim, offset, margin)
    
    if not X_list:
        print("Warning: Coarse sampling did not find any points near obstacles.")
        return np.array([]), np.array([])

    X_coarse = torch.cat(X_list, 0)[:coarse_num]
    Y_coarse = torch.cat(Y_list, 0)[:coarse_num]
    
    # 2. 确定敏感区域
    sensitive_mask = Y_coarse[:, 0] < (margin * 0.2)
    fine_centers = X_coarse[sensitive_mask][:, :dim]
    
    # 3. 细采样阶段
    print("--- Running Fine Sampling Stage ---")
    fine_num = numsamples - X_coarse.shape[0]
    if fine_num > 0 and fine_centers.shape[0] > 0:
        X_list_fine, Y_list_fine = point_append_list_static([], [], fine_centers, triangles_obs, fine_num, dim, offset/2.0, margin/2.0)
        
        # 4. 合并结果
        if X_list_fine:
            X_fine = torch.cat(X_list_fine, 0)[:fine_num]
            Y_fine = torch.cat(Y_list_fine, 0)[:fine_num]
            X_final = torch.cat([X_coarse, X_fine], 0)
            Y_final = torch.cat([Y_coarse, Y_fine], 0)
        else:
            X_final, Y_final = X_coarse, Y_coarse
    else:
        X_final, Y_final = X_coarse, Y_coarse

    sampled_points = X_final.detach().cpu().numpy()
    speed_dist = Y_final.detach().cpu().numpy()
    speed = np.clip(speed_dist, a_min=offset, a_max=margin) / margin
    return sampled_points, speed


# ------------------------------------------------------------------------------
# Section 3: 主入口函数 (保持不变)
# ------------------------------------------------------------------------------

def sample_speed(path, numsamples, dim):
    """【主入口】这个函数会被 preprocess.py 调用。"""
    try:
        out_path = os.path.dirname(path)
        path_name = os.path.splitext(os.path.basename(out_path))[0]
        task_name = out_path.split('/')[2]
        
        out_file = os.path.join(out_path, 'sampled_points.npy')
        if os.path.exists(out_file):
            print(f"Exists: {out_file}")
            return
            
        # 根据任务类型，读取或设置维度和末端执行器信息
        end_effect = None
        if task_name == 'arm':
            with open(os.path.join(out_path, 'dim')) as f:
                lines = f.readlines()
                dim = int(lines[0].strip())
                if len(lines) > 1:
                    end_effect = lines[1].strip()
        
        # 定义采样边界
        limit = 0.5
        if task_name in ['c3d', 'test']:
            margin = limit / 5.0
        elif 'gibson' in task_name:
            margin = limit / 12.0
        else: # arm
            margin = limit / 12.0
        offset = margin / 10.0

        # 读取归一化后的模型文件
        input_file = os.path.join(out_path, os.path.splitext(os.path.basename(path))[0] + '_scaled.off')
        v, f = igl.read_triangle_mesh(input_file)

        start = timer()
        # 根据任务类型，调用不同的采样总调度函数
        if task_name == 'arm':
            print(f"--- Running ARM Sampling for scene: {out_path} ---")
            sampled_points, speed = arm_rand_sample_bound_points(
                numsamples, dim, v, f, offset, margin, out_path, path_name, end_effect
            )
        else:
            # 对于所有非ARM的静态场景，都使用我们新的粗细采样
            print(f"--- Running Coarse-to-Fine STATIC Sampling for scene: {out_path} ---")
            sampled_points, speed = point_rand_sample_bound_points(
                numsamples, dim, v, f, offset, margin
            )
        end = timer()
        print(f"Sampling finished in {end - start:.2f} seconds.")

        if sampled_points.shape[0] == 0:
            print(f"Warning: No valid points were sampled for scene {out_path}. Skipping file generation.")
            return

        # 保存所有生成的.npy文件
        np.save(out_file, sampled_points)
        np.save(os.path.join(out_path, 'speed.npy'), speed)
        np.save(os.path.join(out_path, 'B.npy'), 0.5 * np.random.normal(0, 1, size=(dim, 128)))
        print(f"Successfully saved .npy files for scene {out_path}")

    except Exception as err:
        print(f'处理 {path} 时发生严重错误: {traceback.format_exc()}')