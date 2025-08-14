# ==============================================================================
# 增强版速度场采样脚本 - 专门针对小障碍物（如小石头）检测优化
# 核心改进：
# 1. 三级分层细化采样策略（粗采样 -> 中等采样 -> 超细采样）
# 2. 基于距离梯度的自适应采样密度
# 3. 小障碍物重点关注采样
# 4. 动态采样参数调整
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
# Section 1: 机械臂（arm）专用函数 (保持原有功能)
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
# Section 2: 增强的小障碍物检测采样核心函数
# ------------------------------------------------------------------------------

def point_obstacle_distance(query_points, triangles_obs):
    """核心距离查询函数，使用BVH加速"""
    query_points = query_points.unsqueeze(dim=0)
    bvh = bvh_distance_queries.BVH()
    distances, _, _, _= bvh(triangles_obs, query_points)
    return torch.sqrt(distances).squeeze()

def detect_small_obstacles_regions(sample_points, distances, triangles_obs, threshold_ratio=0.15):
    """检测可能包含小障碍物的区域"""
    # 计算距离变化梯度来识别小障碍物
    if sample_points.shape[0] < 100:
        return sample_points
    
    # 基于局部距离变化检测可能的小障碍物区域
    very_close_mask = distances < threshold_ratio
    if very_close_mask.sum() == 0:
        return torch.empty((0, sample_points.shape[1]), device='cuda')
    
    small_obstacle_centers = sample_points[very_close_mask]
    print(f"Detected {small_obstacle_centers.shape[0]} potential small obstacle points")
    return small_obstacle_centers

def adaptive_sampling_strategy(X_list, Y_list, triangles_obs, numsamples, dim, offset, margin):
    """自适应三级采样策略 - 专门针对小障碍物优化"""
    
    # === 第一阶段：粗采样 - 全局采样捕捉大障碍物 ===
    print("=== 阶段1: 粗采样 (全局大障碍物检测) ===")
    coarse_samples = int(numsamples * 0.4)  # 减少粗采样比例，为细采样腾出空间
    coarse_X, coarse_Y = [], []
    
    coarse_X, coarse_Y = uniform_sampling_stage(
        coarse_X, coarse_Y, triangles_obs, coarse_samples, dim, offset, margin, 
        stage_name="Coarse", batch_multiplier=6
    )
    
    if not coarse_X:
        print("警告：粗采样未找到任何边界点")
        return X_list, Y_list
    
    X_coarse = torch.cat(coarse_X, 0)[:coarse_samples]
    Y_coarse = torch.cat(coarse_Y, 0)[:coarse_samples]
    
    # === 第二阶段：中等采样 - 在中等敏感区域采样 ===
    print("=== 阶段2: 中等采样 (中等敏感区域) ===")
    medium_samples = int(numsamples * 0.35)
    
    # 识别中等敏感区域 (距离 < margin * 0.3)
    medium_sensitive_mask = Y_coarse[:, 0] < (margin * 0.3)
    medium_centers = X_coarse[medium_sensitive_mask][:, :dim]
    
    medium_X, medium_Y = [], []
    if medium_centers.shape[0] > 0:
        medium_X, medium_Y = localized_sampling_stage(
            medium_X, medium_Y, medium_centers, triangles_obs, medium_samples, dim, 
            offset * 0.7, margin * 0.7, stage_name="Medium", perturbation_scale=0.3
        )
    
    # === 第三阶段：超细采样 - 专门针对小障碍物 ===
    print("=== 阶段3: 超细采样 (小障碍物重点检测) ===")
    
    # 合并现有采样点
    all_X = [X_coarse]
    all_Y = [Y_coarse]
    if medium_X:
        X_medium = torch.cat(medium_X, 0)[:medium_samples]
        Y_medium = torch.cat(medium_Y, 0)[:medium_samples]
        all_X.append(X_medium)
        all_Y.append(Y_medium)
    
    combined_X = torch.cat(all_X, 0)
    combined_Y = torch.cat(all_Y, 0)
    
    # 检测可能的小障碍物区域
    ultra_fine_samples = numsamples - combined_X.shape[0]
    small_obstacle_centers = detect_small_obstacles_regions(
        combined_X[:, :dim], combined_Y[:, 0], triangles_obs, threshold_ratio=0.12
    )
    
    ultra_fine_X, ultra_fine_Y = [], []
    if ultra_fine_samples > 0 and small_obstacle_centers.shape[0] > 0:
        ultra_fine_X, ultra_fine_Y = localized_sampling_stage(
            ultra_fine_X, ultra_fine_Y, small_obstacle_centers, triangles_obs, 
            ultra_fine_samples, dim, offset * 0.3, margin * 0.4, 
            stage_name="UltraFine", perturbation_scale=0.1
        )
    
    # === 合并所有阶段的结果 ===
    final_X_list = coarse_X
    final_Y_list = coarse_Y
    
    if medium_X:
        final_X_list.extend(medium_X)
        final_Y_list.extend(medium_Y)
    
    if ultra_fine_X:
        final_X_list.extend(ultra_fine_X)
        final_Y_list.extend(ultra_fine_Y)
    
    X_list.extend(final_X_list)
    Y_list.extend(final_Y_list)
    
    return X_list, Y_list

def uniform_sampling_stage(X_list, Y_list, triangles_obs, num_samples, dim, offset, margin, 
                          stage_name="Uniform", batch_multiplier=8):
    """均匀采样阶段"""
    OutsideSize, WholeSize = num_samples + 2, 0
    
    while OutsideSize > 0:
        batch_size = batch_multiplier * num_samples
        P = torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5
        dP = torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5
        rL = (torch.rand((batch_size, 1), dtype=torch.float32, device='cuda')) * np.sqrt(dim)
        nP = P + torch.nn.functional.normalize(dP, dim=1) * rL
        
        PointsInside = torch.all((nP <= 0.5) & (nP >= -0.5), dim=1) & torch.all((P <= 0.5) & (P >= -0.5), dim=1)
        x0, x1 = P[PointsInside, :], nP[PointsInside, :]
        
        if x0.shape[0] <= 1: 
            continue
        
        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        where_d = (obs_distance0 > offset) & (obs_distance0 < margin)
        
        if where_d.sum() == 0: 
            continue
        
        x0_f, x1_f, y0_f = x0[where_d], x1[where_d], obs_distance0[where_d]
        y1_f = point_obstacle_distance(x1_f, triangles_obs)
        
        X_list.append(torch.cat((x0_f, x1_f), 1))
        Y_list.append(torch.cat((y0_f.unsqueeze(1), y1_f.unsqueeze(1)), 1))
        
        OutsideSize -= x0_f.shape[0]
        WholeSize += x0_f.shape[0]
        print(f"{stage_name} 采样: 已收集 {WholeSize}/{num_samples} 个点...")
        
        if WholeSize >= num_samples: 
            break
    
    return X_list, Y_list

def localized_sampling_stage(X_list, Y_list, centers, triangles_obs, num_samples, dim, 
                           offset, margin, stage_name="Localized", perturbation_scale=0.2):
    """局部化采样阶段 - 在指定中心周围密集采样"""
    if centers.shape[0] == 0:
        return X_list, Y_list
    
    OutsideSize, WholeSize = num_samples + 2, 0
    
    while OutsideSize > 0:
        batch_size = 8 * num_samples
        
        # 随机选择中心点
        center_indices = torch.randint(0, centers.shape[0], (batch_size,), device='cuda')
        selected_centers = centers[center_indices]
        
        # 在中心点周围生成扰动
        perturbation = (torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5) * (perturbation_scale * margin)
        P = selected_centers + perturbation
        
        # 法向量方向采样
        dP = torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5
        rL = (torch.rand((batch_size, 1), dtype=torch.float32, device='cuda')) * perturbation_scale
        nP = P + torch.nn.functional.normalize(dP, dim=1) * rL
        
        # 边界检查
        PointsInside = torch.all((nP <= 0.5) & (nP >= -0.5), dim=1) & torch.all((P <= 0.5) & (P >= -0.5), dim=1)
        x0, x1 = P[PointsInside, :], nP[PointsInside, :]
        
        if x0.shape[0] <= 1: 
            continue
        
        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        where_d = (obs_distance0 > offset) & (obs_distance0 < margin)
        
        if where_d.sum() == 0: 
            continue
        
        x0_f, x1_f, y0_f = x0[where_d], x1[where_d], obs_distance0[where_d]
        y1_f = point_obstacle_distance(x1_f, triangles_obs)
        
        X_list.append(torch.cat((x0_f, x1_f), 1))
        Y_list.append(torch.cat((y0_f.unsqueeze(1), y1_f.unsqueeze(1)), 1))
        
        OutsideSize -= x0_f.shape[0]
        WholeSize += x0_f.shape[0]
        print(f"{stage_name} 采样: 已收集 {WholeSize}/{num_samples} 个点...")
        
        if WholeSize >= num_samples: 
            break
    
    return X_list, Y_list

def point_rand_sample_bound_points_enhanced(numsamples, dim, v, f, offset, margin):
    """增强版静态场景采样 - 三级分层采样专门针对小障碍物优化"""
    numsamples = int(numsamples)
    triangles_obs = torch.tensor(v[f], dtype=torch.float32, device='cuda').unsqueeze(0)
    
    print(f"开始增强版三级采样，目标样本数: {numsamples}")
    print(f"采样参数 - offset: {offset:.4f}, margin: {margin:.4f}")
    
    X_list, Y_list = [], []
    
    # 使用自适应三级采样策略
    X_list, Y_list = adaptive_sampling_strategy(
        X_list, Y_list, triangles_obs, numsamples, dim, offset, margin
    )
    
    if not X_list:
        print("警告: 采样失败，未找到有效的边界点")
        return np.array([]), np.array([])
    
    # 合并所有采样结果
    X_final = torch.cat(X_list, 0)[:numsamples]
    Y_final = torch.cat(Y_list, 0)[:numsamples]
    
    print(f"采样完成！实际采样点数: {X_final.shape[0]}")
    
    # 转换为numpy并计算速度场
    sampled_points = X_final.detach().cpu().numpy()
    speed_dist = Y_final.detach().cpu().numpy()
    speed = np.clip(speed_dist, a_min=offset, a_max=margin) / margin
    
    return sampled_points, speed

# ------------------------------------------------------------------------------
# Section 3: 主入口函数
# ------------------------------------------------------------------------------

def sample_speed(path, numsamples, dim):
    """主入口函数 - 增强版本专门针对小障碍物检测"""
    try:
        out_path = os.path.dirname(path)
        path_name = os.path.splitext(os.path.basename(out_path))[0]
        task_name = out_path.split('/')[2]
        
        print(f"=== 增强版采样启动 ===")
        print(f"场景: {out_path}")
        print(f"任务类型: {task_name}")
        
        out_file = os.path.join(out_path, 'sampled_points.npy')
        if os.path.exists(out_file):
            print(f"文件已存在: {out_file}")
            return
            
        # 根据任务类型设置参数
        end_effect = None
        if task_name == 'arm':
            with open(os.path.join(out_path, 'dim')) as f:
                lines = f.readlines()
                dim = int(lines[0].strip())
                if len(lines) > 1:
                    end_effect = lines[1].strip()
        
        # 采样边界设置 - 针对小障碍物优化
        limit = 0.5
        if task_name in ['c3d', 'test']:
            margin = limit / 8.0  # 减小margin以提高小障碍物敏感性
        elif 'gibson' in task_name:
            margin = limit / 15.0  # 进一步减小gibson场景的margin
        else: # arm
            margin = limit / 12.0
        offset = margin / 15.0  # 减小offset提高精度
        
        print(f"优化采样参数 - margin: {margin:.4f}, offset: {offset:.4f}")
        
        # 读取mesh文件
        input_file = os.path.join(out_path, os.path.splitext(os.path.basename(path))[0] + '_scaled.off')
        v, f = igl.read_triangle_mesh(input_file)
        
        start = timer()
        
        # 根据任务类型选择采样策略
        if task_name == 'arm':
            print("使用机械臂采样策略")
            sampled_points, speed = arm_rand_sample_bound_points(
                numsamples, dim, v, f, offset, margin, out_path, path_name, end_effect
            )
        else:
            # 对于静态场景使用增强版三级采样
            print("使用增强版三级采样策略 (专门针对小障碍物优化)")
            sampled_points, speed = point_rand_sample_bound_points_enhanced(
                numsamples, dim, v, f, offset, margin
            )
        
        end = timer()
        print(f"采样完成，耗时: {end - start:.2f} 秒")
        
        if sampled_points.shape[0] == 0:
            print(f"警告: 场景 {out_path} 未采样到有效点，跳过文件生成")
            return
        
        # 保存结果
        np.save(out_file, sampled_points)
        np.save(os.path.join(out_path, 'speed.npy'), speed)
        np.save(os.path.join(out_path, 'B.npy'), 0.5 * np.random.normal(0, 1, size=(dim, 128)))
        
        print(f"成功保存采样结果到 {out_path}")
        print(f"最终采样点数: {sampled_points.shape[0]}")
        print(f"速度场范围: [{speed.min():.3f}, {speed.max():.3f}]")
        
    except Exception as err:
        print(f'处理 {path} 时发生错误: {traceback.format_exc()}')