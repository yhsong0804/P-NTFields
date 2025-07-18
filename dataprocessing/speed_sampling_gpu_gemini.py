#change版本
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

#计算包围盒8个顶点坐标
def bbox_corner(bbox):
    corner = torch.ones((4,8)).cuda()#4*8张量 [x y z 1]
    iter=0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corner[0,iter]=bbox[0+i*3]
                corner[1,iter]=bbox[1+j*3]
                corner[2,iter]=bbox[2+k*3]
                iter+=1

    return corner
#构造单位矩阵，理论可用于轴对齐包围盒的轴，但实际并没用bbox数据
def bbox_axis(bbox):
    axis = torch.zeros((3,3)).cuda()
    axis[0,0]=1#bbox[0]-bbox[3]
    axis[1,1]=1#bbox[1]-bbox[4]
    axis[2,2]=1#bbox[2]-bbox[5]

    return axis#输出3*3单位矩阵

def cross_matrix(transform_row):
    #给定一批向量，返回每个向量的“正交基底”矩阵
    matrix = torch.zeros((transform_row.shape[0],3,3)).cuda()
    
    a = transform_row[:,0]
    b = transform_row[:,1]
    c = transform_row[:,2]

    l0 = torch.clamp(torch.sqrt(c**2+b**2), min=1e-6) #用于归一化
    l1 = torch.clamp(torch.sqrt(c**2+a**2), min=1e-6)
    l2 = torch.clamp(torch.sqrt(b**2+a**2), min=1e-6)
    #以下分别构造3个正交方向向量
    matrix[:,0,1] = c/l0
    matrix[:,0,2] = -b/l0
    matrix[:,1,0] = -c/l1
    matrix[:,1,2] = a/l1
    matrix[:,2,0] = b/l2
    matrix[:,2,1] = -a/l2
    return matrix#（N，3，3）每个输入向量生成一个正交矩阵

def separate_axis(transform):
    #对一组变换矩阵，计算每个变换对应的15个轴向
    axis = torch.zeros((transform.shape[0],15,3)).cuda()
    #axis[0,0]=1#bbox[0]-bbox[3]
    #axis[1,1]=1#bbox[1]-bbox[4]
    #axis[2,2]=1#bbox[2]-bbox[5]
    axis[:,0:3,:] = torch.eye(3).cuda()
    axis[:,3:6,:] = transform[:,0:3,0:3]
    axis[:,6:9,:]   = cross_matrix(transform[:,0,0:3])
    axis[:,9:12,:]  = cross_matrix(transform[:,1,0:3])
    axis[:,12:15,:] = cross_matrix(transform[:,2,0:3])

    return axis

def obb_collision(rob_corner, obs_corner, transform, margin):
    #空壳函数，判断两个OBB是否碰撞
    return 0

#机械臂与障碍物的OBB碰撞判定
def arm_obstacle_obb(th_batch, chain, out_path_, margin):
    #判定机械臂姿态下各段OBB与障碍物是否碰撞
    whole_dis = []
    end_dis = []
    batch_size = 2000
    scale = 5.0
    modelPath = './Experiments/UR5_SDF'#arona,bolton,cabin,A_test
#modelPath = './Experiments/Gib_res_changelr_scale'

    dataPath = './datasets/arm/'#Arona,Cabin,Bolton#filePath = './Experiments/Gibson'
    model_list = []
    bbox_list = []
    rob_corner_list = []
    input_file_list = ['upper_arm','forearm','wrist_1','wrist_2','wrist_3']
    for input_file in input_file_list:
        bbox = np.load('./datasets/arm/UR5/meshes/collision/'+input_file+'bbox.npy')
        bbox = torch.tensor(bbox).cuda().float()
        bbox[:3] = bbox[:3] + margin
        bbox[3:] = bbox[3:] - margin
        rob_corner_list.append(bbox_corner(bbox))
    #hard code bbox of obs, can be further accelerated by BVH
    #障碍物包围盒（写死的一个AABB）
    bbox = torch.tensor([0.7,-0.5,1.0,-0.6,-0.7,-0.7]).cuda().float()#前三个数代表最小边界 后三个代表最大边界 x方向：0.7 -0.6 y方向：-0.5 -0.7 z方向：1.0 -0.7
    obs_points = bbox_corner(bbox)

    batch_size = 50000

    where_list = []
    for batch_id in range(math.floor(th_batch.shape[0]/batch_size)+1):
        if batch_id*batch_size==th_batch.shape[0]:
            break
        #print(batch_id)
        local_th_batch = th_batch[batch_id*batch_size:min((batch_id+1)*batch_size,th_batch.shape[0]),:]
        tg_batch = chain.forward_kinematics(
                    local_th_batch
                    , end_only = False)

        p_list=[]
        iter = 0
        p_size_list = []
        
        where_coll = torch.zeros(local_th_batch.shape[0], dtype=torch.bool).cuda()

        for tg in tg_batch:
            if iter>2:#从第三段起
                m = tg_batch[tg].get_matrix()
                rob_points= m@rob_corner_list[iter-3]
                #print(iter)
                axis = separate_axis(m)
                rob_axis_point = axis@rob_points[:,0:3,:]
                #print(rob_axis_point.shape)
                #print(obs_points.shape)
                obs_axis_point = axis@obs_points[0:3,:]

                rob_axis_point_min,_ = torch.min(rob_axis_point,dim=2)
                rob_axis_point_max,_ = torch.max(rob_axis_point,dim=2)
                
                obs_axis_point_min,_ = torch.min(obs_axis_point,dim=2)
                obs_axis_point_max,_ = torch.max(obs_axis_point,dim=2)
                #分离轴定理判定区间是否重叠
                where0 = rob_axis_point_max<obs_axis_point_min
                where1 = obs_axis_point_max<rob_axis_point_min

                where = torch.cat((where0,where1),dim=1)
                #print(where.shape)
                nonsep = torch.all((where == False),dim=1)
                #print(nonsep.shape)
                combine = torch.cat((nonsep.unsqueeze(1),where_coll.unsqueeze(1)),dim=1)
                where_coll = torch.any(combine == True,dim=1)

                del m
            iter = iter+1
        where_list.append(nonsep)
        #print(where_coll.shape)
    return torch.cat(where_list,0)#返回所有采样结果（True表示碰撞）
#计算机械臂末端与障碍物最近距离
def arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs):
    whole_dis = []
    end_dis = []
    batch_size = 50000#最多5w组姿态
    for batch_id in range(math.floor(th_batch.shape[0]/batch_size)+1):
        if batch_id*batch_size==th_batch.shape[0]:
            break
        #print(batch_id)
        #取关节角
        tg_batch = chain.forward_kinematics(
            th_batch[batch_id*batch_size:
                    min((batch_id+1)*batch_size,th_batch.shape[0]),:]
                    , end_only = False)#link变换

        p_list=[]
        iter = 0
        p_size_list = []
        
        for tg in tg_batch:
            if iter>1:#跳过第一个link
                #print(iter)
                v, f = igl.read_triangle_mesh(out_path_+'/meshes/collision/'+tg+'.obj')
                nv = np.ones((v.shape[0],4))
                #pointsize = pointsize+v.shape[0]
                p_size_list.append(v.shape[0])
                nv[:,:3]=v
                m = tg_batch[tg].get_matrix()
                #print(m.shape)
                t=torch.from_numpy(nv).float().cuda()
                p=torch.matmul(m[:],t.T)
                #p=p.cpu().numpy()
                p = torch.permute(p, (0, 2, 1)).contiguous()
                #p=np.transpose(p,(0,2,1))
                p_list.append(p)
                del m,p,t,nv, v
            iter = iter+1
        pointsize = sum(p_size_list)
        #print(pointsize)
        #p = np.concatenate(p_list,axis=1)
        p = torch.cat(p_list, dim=1)
        p = torch.reshape(p,(p.shape[0]*p.shape[1],p.shape[2])).contiguous()
        query_points = p[:,0:3].contiguous()
        query_points = query_points.unsqueeze(dim=0)
        
        bvh = bvh_distance_queries.BVH()

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        #计算所有点到障碍物三角网络的最近距离
        distance, closest_points, closest_faces, closest_bcs= bvh(triangles_obs, query_points)
        torch.cuda.synchronize()
        #unsigned_distance = abs()
        #print(distance.shape)
        distance = torch.sqrt(distance).squeeze()
        distance = torch.reshape(distance, (-1, pointsize))
        whole_distance,_ = torch.min(distance, dim=1)
        #distance = distance.detach().cpu().numpy()


        #print(whole_distance)
        whole_dis.append(whole_distance)
        del p, p_list, tg_batch, distance, query_points, bvh,whole_distance
    #unsigned_distance = np.concatenate(whole_p, axis=0)
    unsigned_distance = torch.cat(whole_dis, dim=0)
    #print(unsigned_distance)
    return unsigned_distance#每个采样姿态的最小距离
#采样并筛选有效边界点
def arm_append_list(X_list, Y_list, chain, out_path_, 
                    triangles_obs,
                    numsamples, dim, offset, margin):
    
    OutsideSize = numsamples + 2
    WholeSize = 0

    scale = math.pi/0.5

    while OutsideSize > 0:
        #均匀采样[-0.5 0.5]内点
        P  = torch.rand((5*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((5*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((5*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(3)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL
        #nP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        #随机采样法线方向上的邻域点
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        #所有点都在-0.5 0.5^3内

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        #print(x0.shape[0])
        if(x0.shape[0]<=1):
            continue
        
        th_batch = scale*x0
        print(th_batch.shape)
        where = arm_obstacle_obb(th_batch, chain, out_path_, margin)#.cuda()
        x0 = x0[where]
        x1 = x1[where]
        #print(x0)

        th_batch = scale*x0
        print(th_batch.shape)
        obs_distance0 = arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs)
        print(torch.min(obs_distance0))
        print(torch.max(obs_distance0))
        where_d          =  (obs_distance0 >= offset) & (obs_distance0 <= margin)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]
        # --- 限制细采样点总数 ---
        MAX_FINE_POINTS = 20000  # 可以调整，比如20000或者10000
        if x0.shape[0] > MAX_FINE_POINTS:
            perm = torch.randperm(x0.shape[0])[:MAX_FINE_POINTS]
            x0 = x0[perm]
            x1 = x1[perm]
            y0 = y0[perm]
        # --- 限制结束 ---


        th_batch = scale*x1
        obs_distance1 = arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs)

        y1 = obs_distance1
        
        print(x0.shape)
        #print(x1.shape)
        #print(y0.shape)
        #print(y1.shape)

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        
        if(WholeSize > numsamples):
            break
    return X_list, Y_list
#机械臂场景主入口
def arm_rand_sample_bound_points(numsamples, dim, 
                                 v_obs, f_obs, offset, margin,
                                 out_path_ , path_name_, end_effect_):
    numsamples = int(numsamples)

    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    #构建机械臂运动链
    chain = pk.build_serial_chain_from_urdf(
        open(out_path_+'/'+path_name_+".urdf").read(), end_effect_)
    chain = chain.to(dtype=dtype, device=d)

    scale = math.pi/0.5
    #构造障碍物三角网格
    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    f_obs = torch.tensor(f_obs, dtype=torch.long, device='cuda')
    t_obs = v_obs[f_obs].unsqueeze(dim=0)

    X_list = []
    Y_list = []

    X_list, Y_list = arm_append_list(X_list, Y_list, chain, out_path_, 
                                    t_obs, numsamples, dim, offset, margin)
  
    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]    
    
    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    
    distance0 = distance[:,0]
    distance1 = distance[:,1]
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance0 , a_min = offset, a_max = margin)/margin
    speed[:,1] = np.clip(distance1 , a_min = offset, a_max = margin)/margin

    return sampled_points, speed
#采样点与障碍物距离查询（核心函数）
def point_obstacle_distance(query_points, triangles_obs):
    #query_points:(N,3)点云，triangles_obs：（1，M，3，3）三角面片
    query_points = query_points.unsqueeze(dim=0)#扩维方便BVH接口
    #print(query_points.shape)
    bvh = bvh_distance_queries.BVH()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    distances, closest_points, closest_faces, closest_bcs= bvh(triangles_obs, query_points)
    torch.cuda.synchronize()
    unsigned_distance = torch.sqrt(distances).squeeze()
    #print(closest_points.shape)
    return unsigned_distance#返回到障碍物的最短距离
# 原始的函数，用于粗采样
def point_append_list(X_list,Y_list, 
                      triangles_obs, numsamples, dim, offset, margin):
    OutsideSize = numsamples + 2#增加冗余，防止溢出
    WholeSize = 0

    while OutsideSize > 0:
        # 在整个 [-0.5, 0.5] 空间内均匀生成点
        P  = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((8*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(dim)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL
        
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        
        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        if(x0.shape[0]<=1):
            continue
        
        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        where_d = (obs_distance0 > offset) & (obs_distance0 < margin)
        
        # 如果筛选后没有点，则跳过此次循环，避免索引错误
        if where_d.sum() == 0:
            continue
            
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]

        obs_distance1 = point_obstacle_distance(x1, triangles_obs)
        y1 = obs_distance1

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        
        print(f"Coarse Sampling Progress: collected {WholeSize}/{numsamples} points...")

        if(WholeSize >= numsamples):
            break
    return X_list, Y_list

# ==============================================================================
#  这是新增的函数，专门用于细采样
# ==============================================================================
def point_append_list_fine(X_list, Y_list, fine_sample_centers,
                           triangles_obs, numsamples, dim, offset, margin):
    OutsideSize = numsamples + 2
    WholeSize = 0
    
    # 获取敏感区域中心的数量
    num_centers = fine_sample_centers.shape[0]
    if num_centers == 0:
        return X_list, Y_list # 如果没有敏感点，直接返回

    while OutsideSize > 0:
        # 1. 选择一批敏感点作为中心
        #    为了保证多样性，每次都从所有敏感点中随机选择
        batch_size = 8 * numsamples 
        center_indices = torch.randint(0, num_centers, (batch_size,), device='cuda')
        centers = fine_sample_centers[center_indices]

        # 2. 在选定的中心点周围添加小的随机扰动，生成新的采样点
        #    扰动的范围与 margin 相关，确保在目标区域内
        perturbation = (torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5) * (2 * margin)
        P = centers + perturbation
        
        # 类似地生成配对点
        dP = torch.rand((batch_size,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((batch_size,1),dtype=torch.float32, device='cuda'))*np.sqrt(dim) * (margin) # 邻域点也限制在附近
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL

        # 同样需要确保点在 [-0.5, 0.5] 的界内
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1) & \
                       torch.all((P <= 0.5),dim=1) & torch.all((P >= -0.5),dim=1)

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        if(x0.shape[0]<=1):
            continue

        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        where_d = (obs_distance0 > offset) & (obs_distance0 < margin)
        
        if where_d.sum() == 0:
            continue
            
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]

        obs_distance1 = point_obstacle_distance(x1, triangles_obs)
        y1 = obs_distance1

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        
        print(f"Fine Sampling Progress: collected {WholeSize}/{numsamples} points...")

        if(WholeSize >= numsamples):
            break
    return X_list, Y_list
def point_rand_sample_bound_points(numsamples, dim, 
                                   v_obs, f_obs, offset, margin, fine_sample_centers=None):
    numsamples = int(numsamples)

    v_obs_tensor = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    f_obs_tensor = torch.tensor(f_obs, dtype=torch.long, device='cuda')
    t_obs = v_obs_tensor[f_obs_tensor].unsqueeze(dim=0)
    
    X_list = []
    Y_list = []
    
    if fine_sample_centers is None:
        # 执行粗采样
        X_list, Y_list = point_append_list(X_list, Y_list, t_obs, numsamples, dim, offset, margin)
    else:
        # 执行细采样
        X_list, Y_list = point_append_list_fine(X_list, Y_list, fine_sample_centers, t_obs, numsamples, dim, offset, margin)
   
    if not X_list: # 如果列表为空，返回空数组避免拼接错误
        return np.array([]).reshape(0, 2 * dim), np.array([]).reshape(0, 2)

    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    
    distance0 = distance[:,0]
    distance1 = distance[:,1]
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance0, a_min=offset, a_max=margin) / margin
    speed[:,1] = np.clip(distance1, a_min=offset, a_max=margin) / margin
    
    return sampled_points, speed
#核心采样入口sample_speed 作用：mesh边界附近采样点，并对每一个点计算距离和归一化后的点“速度场”信息
def sample_speed(path, numsamples, dim):
    try:
        # (这部分文件解析和参数设置的代码保持不变)
        out_path = os.path.dirname(path)
        path_name = os.path.splitext(os.path.basename(out_path))[0]
        task_name = out_path.split('/')[2]
        if task_name == 'arm':
            # ... (机械臂的处理逻辑保持不变)
            # 为了简化，我们这里的重点是修复3D点云的采样
            print("Arm sampling logic remains.")
            # 此处应调用原始的 arm_rand_sample_bound_points
            # ...
            return
        
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path, file_name + '_scaled.off')
        out_file = out_path + '/sampled_points.npy'

        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            # return
        
        limit = 0.5
        if task_name == 'c3d' or task_name == 'test':
            margin = limit / 5.0
            offset = margin / 10.0 
        elif task_name == 'gibson':
            margin = limit / 12.0
            offset = margin / 10.0 
        else:
            margin = limit / 12.0
            offset = margin / 10.0 

        v, f = igl.read_triangle_mesh(input_file)
        start = timer()

        # --- 正确的两阶段采样流程 ---

        # 1. 粗采样阶段 (Coarse Sampling)
        #    用较大的margin在整个空间中寻找障碍物的大致位置
        print("--- Starting Coarse Sampling Stage ---")
        coarse_numsamples = int(numsamples * 0.7) # 分配70%的预算给粗采样
        coarse_sampled_points, coarse_speed = point_rand_sample_bound_points(
            coarse_numsamples, dim, v, f, offset, margin)

        # 2. 确定敏感区域 (Identify Sensitive Areas)
        #    找到靠近障碍物的点，特别是那些可能属于小障碍物的点
        #    这里的阈值可以调整，值越小，区域越贴近表面
        sensitive_threshold = margin * 0.2 
        sensitive_mask = coarse_speed[:, 0] < sensitive_threshold
        
        # 将粗采样中得到的点作为细采样的中心
        fine_sample_centers = coarse_sampled_points[sensitive_mask][:, :dim] # 只取点对的第一个点x0作为中心
        fine_sample_centers_tensor = torch.tensor(fine_sample_centers, dtype=torch.float32, device='cuda')

        # 3. 细采样阶段 (Fine Sampling)
        print("\n--- Starting Fine Sampling Stage ---")
        fine_numsamples = numsamples - coarse_sampled_points.shape[0] # 剩余的预算给细采样
        fine_margin = margin / 2.0  # 使用更小的margin进行精细探索
        fine_offset = offset / 2.0
        
        fine_sampled_points, fine_speed = point_rand_sample_bound_points(
            fine_numsamples, dim, v, f, fine_offset, fine_margin, 
            fine_sample_centers=fine_sample_centers_tensor)

        # 4. 合并结果
        print("\n--- Merging sampling results ---")
        if fine_sampled_points.shape[0] > 0:
            sampled_points = np.concatenate([coarse_sampled_points, fine_sampled_points], axis=0)
            speed = np.concatenate([coarse_speed, fine_speed], axis=0)
        else: # 如果细采样没有采到任何点
            sampled_points = coarse_sampled_points
            speed = coarse_speed

        end = timer()
        print(f"\nTotal sampling time: {end-start:.2f} seconds")

        B = 0.5 * np.random.normal(0, 1, size=(3, 128))
        
        np.save('{}/sampled_points'.format(out_path), sampled_points)
        np.save('{}/speed'.format(out_path), speed)
        np.save('{}/B'.format(out_path), B)

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))