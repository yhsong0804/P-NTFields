# ==============================================================================
# 最终版本：智能采样脚本 speed_sampling_gpu.py
# 功能：
# 1. 自动检测场景是静态还是动态。
# 2. 对静态场景，使用“粗-细”两阶段采样，以精确捕捉小障碍物。
# 3. 对动态场景，使用“时空”采样，生成带时间戳的数据。
# 4. 移除了当前任务无关的复杂机械臂运动学代码，使逻辑更清晰。
# ==============================================================================

import os 
import glob
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch

# 我们把bvh的import放在这里，因为它只在GPU可用时才需要
# 这是一个专门用于GPU加速计算点到网格距离的库
try:
    import bvh_distance_queries
except ImportError:
    print("警告: bvh_distance_queries 未安装，GPU加速采样将不可用。")

# ##############################################################################
# Section 1: 静态场景采样核心函数 (用于解决小石头问题)
# ##############################################################################

def point_obstacle_distance(query_points, triangles_obs):
    """
    【核心工具函数】使用BVH在GPU上并行计算大量查询点到障碍物表面的最短距离。
    
    Args:
        query_points (Tensor): 需要查询的点云，shape: [N, 3]。
        triangles_obs (Tensor): 障碍物的三角面片，shape: [1, M, 3, 3]。
        
    Returns:
        Tensor: 每个查询点到障碍物的距离，shape: [N,]。
    """
    # bvh库要求输入维度为 [1, N, 3]，所以需要增加一个维度
    query_points = query_points.unsqueeze(dim=0)
    bvh = bvh_distance_queries.BVH()
    # bvh()返回的是距离的平方，所以需要开根号
    distances, _, _, _= bvh(triangles_obs, query_points)
    return torch.sqrt(distances).squeeze()

def point_append_list_static(X_list, Y_list, fine_sample_centers, triangles_obs, numsamples, dim, offset, margin):
    """
    【静态场景采样循环】在一个循环中不断生成和筛选点，直到满足数量要求。
    这个函数是“双模”的，既可以进行全局的粗采样，也可以进行局部的细采样。

    Args:
        fine_sample_centers (Tensor or None): 如果是细采样，这里传入粗采样找到的敏感点作为中心；
                                             如果是粗采样，则为None。
    """
    is_fine_sampling = fine_sample_centers is not None
    OutsideSize = numsamples + 2
    WholeSize = 0
    
    num_centers = 0
    if is_fine_sampling:
        num_centers = fine_sample_centers.shape[0]
        # 如果没有找到任何敏感点，就没必要进行细采样了
        if num_centers == 0: return X_list, Y_list

    while OutsideSize > 0:
        batch_size = 8 * numsamples  # 为了效率，一次生成远超需求的点，再进行筛选

        if is_fine_sampling:
            # --- 细采样逻辑 ---
            # 1. 从敏感点中心随机选择一批
            center_indices = torch.randint(0, num_centers, (batch_size,), device='cuda')
            centers = fine_sample_centers[center_indices]
            # 2. 在中心点周围添加一个小的、与margin相关的扰动来生成新点
            perturbation = (torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5) * (2 * margin)
            P = centers + perturbation
        else:
            # --- 粗采样逻辑 ---
            # 在整个 [-0.5, 0.5] 的归一化空间内均匀生成随机点
            P  = torch.rand((batch_size,dim),dtype=torch.float32, device='cuda')-0.5
        
        # 为每个点P，在它附近再生成一个配对点nP
        dP = torch.rand((batch_size,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((batch_size,1),dtype=torch.float32, device='cuda'))*np.sqrt(dim)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL

        # 确保所有生成的点都在归一化空间内
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]
        
        if x0.shape[0] <= 1: continue

        # 计算所有起始点x0到障碍物的距离
        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        # 只保留那些距离在(offset, margin)范围内的点，这些点对学习最有价值
        where_d = (obs_distance0 > offset) & (obs_distance0 < margin)
        
        if where_d.sum() == 0: continue
            
        # 筛选出合格的点
        x0_f = x0[where_d]
        x1_f = x1[where_d]
        y0_f = obs_distance0[where_d]
        # 为合格的配对点也计算距离
        y1_f = point_obstacle_distance(x1_f, triangles_obs)
        
        # 将合格的点对(x0,x1)和对应的距离(y0,y1)存入列表
        X_list.append(torch.cat((x0_f, x1_f), 1))
        Y_list.append(torch.cat((y0_f.unsqueeze(1), y1_f.unsqueeze(1)), 1))
        
        OutsideSize -= x0_f.shape[0]
        WholeSize += x0_f.shape[0]
        stage = "Fine" if is_fine_sampling else "Coarse"
        print(f"Static {stage} Sampling: collected {WholeSize}/{numsamples} points...")
        if WholeSize >= numsamples: break
        
    return X_list, Y_list

def point_rand_sample_bound_points_static(numsamples, dim, v, f, offset, margin):
    """【静态场景总调度】执行完整的“粗-细”两阶段采样流程。"""
    
    # 将numpy格式的mesh数据转换为GPU上的Tensor
    triangles_obs = torch.tensor(v[f], dtype=torch.float32, device='cuda').unsqueeze(0)
    
    # 1. 粗采样阶段：分配70%的预算进行全局搜索
    print("--- Running Coarse Sampling Stage ---")
    coarse_num = int(numsamples * 0.7)
    X_list, Y_list = point_append_list_static([], [], None, triangles_obs, coarse_num, dim, offset, margin)
    
    if not X_list: # 如果粗采样没采到任何点，直接返回空
        return np.array([]), np.array([])

    X_coarse = torch.cat(X_list, 0)[:coarse_num]
    Y_coarse = torch.cat(Y_list, 0)[:coarse_num]
    
    # 2. 确定敏感区域：找到那些离障碍物非常近的点
    sensitive_mask = Y_coarse[:, 0] < (margin * 0.2)
    fine_centers = X_coarse[sensitive_mask][:, :dim] # 只取x0作为细采样中心
    
    # 3. 细采样阶段：用剩余的预算在敏感区域周围进行高密度采样
    print("--- Running Fine Sampling Stage ---")
    fine_num = numsamples - X_coarse.shape[0]
    if fine_num > 0 and fine_centers.shape[0] > 0:
        X_list_fine, Y_list_fine = point_append_list_static([], [], fine_centers, triangles_obs, fine_num, dim, offset/2.0, margin/2.0)
        
        # 4. 合并粗、细两次采样的结果
        if X_list_fine:
            X_fine = torch.cat(X_list_fine, 0)[:fine_num]
            Y_fine = torch.cat(Y_list_fine, 0)[:fine_num]
            X_final = torch.cat([X_coarse, X_fine], 0)
            Y_final = torch.cat([Y_coarse, Y_fine], 0)
        else: # 如果细采样没采到点，就只用粗采样的结果
            X_final, Y_final = X_coarse, Y_coarse
    else:
        X_final, Y_final = X_coarse, Y_coarse

    # 将GPU上的Tensor转回CPU上的Numpy数组，并计算最终的速度值
    sampled_points = X_final.detach().cpu().numpy()
    speed_dist = Y_final.detach().cpu().numpy()
    speed = np.clip(speed_dist, a_min=offset, a_max=margin) / margin
    return sampled_points, speed


# ##############################################################################
# Section 2: 动态场景采样核心函数 (用于解决移动障碍物)
# ##############################################################################

def point_rand_sample_bound_points_dynamic(numsamples, dim, all_meshes, offset, margin):
    """【动态场景总调度】为动态障碍物生成带时间戳的采样数据。"""

    # 将所有时刻的mesh都加载到GPU上
    triangles_obs_list = [torch.tensor(m['v'][m['f']], dtype=torch.float32, device='cuda').unsqueeze(0) for m in all_meshes]
    num_timesteps = len(triangles_obs_list)
    OutsideSize, WholeSize, X_list, Y_list, T_list = numsamples + 2, 0, [], [], []

    while OutsideSize > 0:
        # 为了效率，一次性生成大量候选点
        P = torch.rand((8 * numsamples, dim), dtype=torch.float32, device='cuda') - 0.5
        # 只保留在归一化空间内的点
        x0 = P[torch.all((P <= 0.5) & (P >= -0.5), dim=1)]
        if x0.shape[0] <= 1: continue
        
        # 【核心逻辑】为本批次的每个点，随机分配一个时间戳 t (0, 1, 2, or 3)
        t = torch.randint(0, num_timesteps, (x0.shape[0],), device='cuda')
        
        obs_distance0 = torch.zeros(x0.shape[0], device='cuda')
        
        # 按时刻分组，高效地并行计算距离
        for i in range(num_timesteps):
            time_mask = (t == i) # 找到所有时间戳为 i 的点
            if time_mask.sum() > 0:
                # 计算这些点到 t=i 时刻障碍物的距离
                obs_distance0[time_mask] = point_obstacle_distance(x0[time_mask], triangles_obs_list[i])
        
        # 筛选出距离在(offset, margin)范围内的点
        where_d = (obs_distance0 > offset) & (obs_distance0 < margin)
        if where_d.sum() == 0: continue
            
        # 保存所有合格的数据：点的位置、距离、还有【时间戳】
        x0_f = x0[where_d]
        t_f = t[where_d].unsqueeze(1)
        y0_f = obs_distance0[where_d].unsqueeze(1)
        
        # 为了简化数据结构，我们只采样一个点(x0)，它的配对点(x1)和对应的速度
        # 可以在网络中或数据加载时再处理，这里我们先保证核心信息被保存
        X_list.append(x0_f); Y_list.append(y0_f); T_list.append(t_f)
        
        OutsideSize -= x0_f.shape[0]
        WholeSize += x0_f.shape[0]
        print(f"Dynamic Sampling Progress: collected {WholeSize}/{numsamples} points...")
        if WholeSize >= numsamples: break

    # 将列表中的所有Tensor合并成一个大Tensor
    X, Y, T = [torch.cat(lst, 0)[:numsamples] for lst in (X_list, Y_list, T_list)]
    
    # 构造最终输出的Numpy数组，以匹配原始代码的数据格式
    # sampled_points 需要是 (N, 2*dim)
    sampled_points = np.zeros((numsamples, dim * 2))
    sampled_points[:,:dim] = X.detach().cpu().numpy() # x0
    sampled_points[:,dim:] = X.detach().cpu().numpy() # 用x0填充x1，因为我们只采样了一个点

    # speed 需要是 (N, 2)
    speed = np.zeros((numsamples, 2))
    speed_vals = np.clip(Y.detach().cpu().numpy(), a_min=offset, a_max=margin) / margin
    speed[:, 0] = speed_vals.squeeze()
    speed[:, 1] = speed_vals.squeeze()

    return sampled_points, speed, T.detach().cpu().numpy()


# 用这个最终版本的 sample_speed 函数替换旧的
def sample_speed(path, numsamples, dim):
    try:
        out_path = os.path.dirname(path)
        print(f"\n--- Processing scene: {out_path} ---")

        dynamic_mesh_files = sorted(glob.glob(os.path.join(out_path, 'mesh_t*_scaled.off')))
        is_dynamic = bool(dynamic_mesh_files)

        # --- 【最终修正】: 我们检查所有核心文件是否存在 ---
        points_file = os.path.join(out_path, 'sampled_points.npy')
        speed_file = os.path.join(out_path, 'speed.npy')
        timestamps_file = os.path.join(out_path, 'timestamps.npy')

        # 只有当场景是动态，并且所有三个文件都存在时，才跳过
        if is_dynamic and os.path.exists(points_file) and os.path.exists(speed_file) and os.path.exists(timestamps_file):
            print(f"All dynamic .npy files already exist for scene {out_path}. Skipping.")
            return
        # 如果是静态，并且两个文件存在，就跳过
        elif not is_dynamic and os.path.exists(points_file) and os.path.exists(speed_file):
             print(f"All static .npy files already exist for scene {out_path}. Skipping.")
             return
        # --- 修正结束 ---

        limit = 0.5
        margin = limit / (12.0 if 'gibson' in out_path else 5.0)
        offset = margin / 10.0

        start_time = timer()
        if is_dynamic:
            print("--- Detected Dynamic Scene. Running Spatio-Temporal Sampling. ---")
            all_meshes = [{'v': igl.read_triangle_mesh(f)[0], 'f': igl.read_triangle_mesh(f)[1]} for f in dynamic_mesh_files]
            points, speed, timestamps = point_rand_sample_bound_points_dynamic(numsamples, dim, all_meshes, offset, margin)

            np.save(timestamps_file, timestamps) # 使用我们之前定义的路径
            print("Successfully saved timestamps.npy")
        else:
            print("--- Detected Static Scene. Running Coarse-to-Fine Sampling. ---")
            input_file = os.path.join(out_path, os.path.splitext(os.path.basename(path))[0] + '_scaled.off')
            v, f = igl.read_triangle_mesh(input_file)
            points, speed = point_rand_sample_bound_points_static(numsamples, dim, v, f, offset, margin)

        np.save(points_file, points)
        np.save(speed_file, speed)
        np.save(os.path.join(out_path, 'B.npy'), 0.5 * np.random.normal(0, 1, size=(dim, 128)))

        end_time = timer()
        print(f"Successfully saved .npy files for scene {out_path}. Total time: {end_time - start_time:.2f}s")

    except Exception:
        print(f'处理 {path} 时发生严重错误: {traceback.format_exc()}')