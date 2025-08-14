# 手动分配
# ==============================================================================
# gemini_ICRA最终版v2：基于预算分配的自适应采样算法 speed_sampling_gpu.py
# 核心创新点：
# 1. 勘探与分类：不再边采样边决策，而是先进行一次全局勘探，找到所有潜在的
#    “种子点”，并根据它们靠近的物体ID进行精确分类。
# 2. 预算分配：为不同类别的障碍物（小石头、大石头）设定独立的采样“预算”，
#    从概率影响升级为精确的数量控制。
# 3. 按预算执行：根据分配好的预算，在各种子周围生成精确数量的密集采样点，
#    实现对小石头等关键区域的强力聚焦，同时抑制在大石头上的冗余采样。
# ==============================================================================

import os
import numpy as np
import igl
import traceback
import math
import torch
import open3d as o3d
import pytorch_kinematics as pk
import bvh_distance_queries
from timeit import default_timer as timer

# ==============================================================================
# Section 1: 机械臂专用函数 (保持不变)
# ==============================================================================
# ... (从 bbox_corner 到 arm_rand_sample_bound_points 的所有函数保持不变) ...
# 为了简洁，这里省略这部分代码，请在你本地保留这部分不变。

def bbox_corner(bbox):
    # (此函数内容保持不变)
    pass

def arm_rand_sample_bound_points(numsamples, dim, v_obs, f_obs, offset, margin, out_path_ , path_name_, end_effect_):
    # (此函数内容保持不变)
    pass


# ==============================================================================
# Section 2: 静态场景自适应采样核心函数 (全新重构)
# ==============================================================================

def uniform_random_sampling(num_points, triangles_obs, dim):
    """
    【模块C: 自由空间探索者】
    在整个空间内进行均匀随机采样，主要目的是获取自由空间中的点。
    """
    print(f"--- [Uniform Sampling] Generating {num_points} points for free-space exploration... ---")
    X_list, Y_list = [], []
    collected_points = 0
    bvh = bvh_distance_queries.BVH()

    while collected_points < num_points:
        batch_size = 8 * (num_points - collected_points)
        P  = torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5
        dP = torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5
        rL = (torch.rand((batch_size, 1), dtype=torch.float32, device='cuda')) * np.sqrt(dim)
        nP = P + torch.nn.functional.normalize(dP, dim=1) * rL
        
        PointsInside = torch.all((nP <= 0.5) & (nP >= -0.5), dim=1) & torch.all((P <= 0.5) & (P >= -0.5), dim=1)
        x0, x1 = P[PointsInside,:], nP[PointsInside,:]
        if x0.shape[0] == 0: continue

        y0 = torch.sqrt(bvh(triangles_obs, x0.unsqueeze(0))[0].squeeze())
        y1 = torch.sqrt(bvh(triangles_obs, x1.unsqueeze(0))[0].squeeze())
        
        X_list.append(torch.cat((x0, x1), 1))
        Y_list.append(torch.cat((y0.unsqueeze(1), y1.unsqueeze(1)), 1))
        collected_points += x0.shape[0]
        
    X_final = torch.cat(X_list, 0)[:num_points]
    Y_final = torch.cat(Y_list, 0)[:num_points]
    return X_final, Y_final

def precompute_and_categorize_seeds(triangles_obs, num_exploration_points, dim, offset, margin):
    """
    【模块A: 勘探与分类】
    对整个空间进行一次大规模勘探，找到所有潜在的种子点，并按其最近的物体进行分类。
    """
    print("--- [Exploration Phase] Exploring scene to find all potential seeds... ---")
    
    # 1. 大规模随机撒点进行勘探
    P = torch.rand((num_exploration_points, dim), dtype=torch.float32, device='cuda') - 0.5
    
    # 2. 计算所有点到场景的距离和最近面片索引
    bvh = bvh_distance_queries.BVH()
    squared_distances, _, closest_faces, _ = bvh(triangles_obs, P.unsqueeze(0))
    distances = torch.sqrt(squared_distances.squeeze())
    closest_faces = closest_faces.squeeze().long()

    # 3. 筛选出所有在有效距离范围内的“潜在种子点”
    potential_seeds_mask = (distances > offset) & (distances < margin)
    
    seed_points = P[potential_seeds_mask]
    seed_face_indices = closest_faces[potential_seeds_mask]
    
    print(f"--- [Exploration Phase] Found {len(seed_points)} potential seeds in total. ---")
    return seed_points, seed_face_indices


def budgeted_fine_sampling(seed_points_by_category, budget_by_category, triangles_obs, dim, offset, margin):
    """
    【模块B: 按预算精确采样】
    根据分配好的预算，在各类种子周围精确地生成指定数量的样本点。
    """
    X_final_list, Y_final_list = [], []

    for category_id, seeds in seed_points_by_category.items():
        budget = budget_by_category.get(category_id, 0) # 如果没有为该类别设置预算，则预算为0
        if budget == 0 or len(seeds) == 0:
            continue
            
        print(f"--- [Budgeted Sampling] Generating {budget} points for Category ID: {category_id}... ---")
        
        num_generated = 0
        bvh = bvh_distance_queries.BVH()

        while num_generated < budget:
            # 在该类别的种子上进行扰动，生成新点
            batch_size = 8 * (budget - num_generated) # 过采样
            center_indices = torch.randint(0, len(seeds), (batch_size,), device='cuda')
            centers = seeds[center_indices]
            perturbation = (torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5) * margin
            P = centers + perturbation
            
            # 生成配对点
            dP = torch.rand((batch_size, dim), dtype=torch.float32, device='cuda') - 0.5
            rL = (torch.rand((batch_size, 1), dtype=torch.float32, device='cuda')) * np.sqrt(dim)
            nP = P + torch.nn.functional.normalize(dP, dim=1) * rL
            
            # 筛选
            PointsInside = torch.all((nP <= 0.5) & (nP >= -0.5), dim=1) & torch.all((P <= 0.5) & (P >= -0.5), dim=1)
            x0, x1 = P[PointsInside,:], nP[PointsInside,:]
            if x0.shape[0] == 0: continue
            
            # 计算距离并再次筛选
            dists = torch.sqrt(bvh(triangles_obs, x0.unsqueeze(0))[0].squeeze())
            valid_mask = (dists > offset) & (dists < margin)
            
            x0_f, x1_f, y0_f = x0[valid_mask], x1[valid_mask], dists[valid_mask]
            if x0_f.shape[0] == 0: continue

            y1_f = torch.sqrt(bvh(triangles_obs, x1_f.unsqueeze(0))[0].squeeze())
            
            X_final_list.append(torch.cat((x0_f, x1_f), 1))
            Y_final_list.append(torch.cat((y0_f.unsqueeze(1), y1_f.unsqueeze(1)), 1))
            
            num_generated += x0_f.shape[0]

    return X_final_list, Y_final_list


def point_rand_sample_bound_points(numsamples, dim, v, f, offset, margin, mesh):
    """
    【自适应采样总调度 - 混合策略版】
    """
    numsamples = int(numsamples)
    triangles_obs = torch.tensor(v[f], dtype=torch.float32, device='cuda').unsqueeze(0)

    # 1. --- 制定混合策略预算 ---
    uniform_budget = int(numsamples * 0.4) # 40%的预算用于自由空间探索
    adaptive_budget = numsamples - uniform_budget # 60%的预算用于关键区域聚焦
    
    # 2. --- 执行【自由空间探索】 ---
    X_uniform, Y_uniform = uniform_random_sampling(uniform_budget, triangles_obs, dim)

    # 3. --- 执行【关键区域聚焦】 ---
    # 3a. 勘探与分类
    num_exploration_points = adaptive_budget * 4 # 使用4倍自适应预算的点进行勘探
    potential_seeds, seed_face_indices = precompute_and_categorize_seeds(triangles_obs, num_exploration_points, dim, offset, margin)
    
    X_adaptive, Y_adaptive = None, None
    if len(potential_seeds) > 0:
        triangle_clusters, _, _ = mesh.cluster_connected_triangles()
        face_to_cluster_id = torch.from_numpy(np.asarray(triangle_clusters)).cuda().long()
        seed_cluster_ids = face_to_cluster_id[seed_face_indices]
        
        seed_points_by_category = {}
        for cluster_id in torch.unique(seed_cluster_ids):
            mask = seed_cluster_ids == cluster_id
            seed_points_by_category[cluster_id.item()] = potential_seeds[mask]
            
        # 3b. 制定自适应部分的内部预算
        adaptive_budget_allocation = {
            2:    int(adaptive_budget * 0.7),  # 小石头 (ID=2)，分配自适应预算的70%
            1339: int(adaptive_budget * 0.15), # 大石头A (ID=1339)，分配15%
            1743: int(adaptive_budget * 0.15), # 大石头B (ID=1743)，分配15%
        }
        
        # 3c. 按预算执行细采样
        X_list_adaptive, Y_list_adaptive = budgeted_fine_sampling(seed_points_by_category, adaptive_budget_allocation, triangles_obs, dim, offset, margin)

        if X_list_adaptive:
            X_adaptive = torch.cat(X_list_adaptive, 0)
            Y_adaptive = torch.cat(Y_list_adaptive, 0)
            # 对自适应采样结果进行下采样，确保不超过其预算
            if len(X_adaptive) > adaptive_budget:
                indices = torch.randperm(len(X_adaptive))[:adaptive_budget]
                X_adaptive, Y_adaptive = X_adaptive[indices], Y_adaptive[indices]

    # 4. --- 合并最终结果 ---
    if X_adaptive is not None:
        X_final = torch.cat([X_uniform, X_adaptive], 0)
        Y_final = torch.cat([Y_uniform, Y_adaptive], 0)
    else:
        print("警告：自适应采样阶段未能生成任何点，最终结果只包含均匀采样。")
        X_final, Y_final = X_uniform, Y_uniform


    sampled_points = X_final.detach().cpu().numpy()
    speed_dist = Y_final.detach().cpu().numpy()
    speed = np.clip(speed_dist, a_min=offset, a_max=margin) / margin
    return sampled_points, speed

# ==============================================================================
# Section 3: 主入口函数 (已修改以支持新流程)
# ==============================================================================

def sample_speed(path, numsamples, dim):
    """【主入口】这个函数会被 preprocess.py 调用。"""
    try:
        out_path = os.path.dirname(path)
        task_name = out_path.split('/')[2]

        # (省略了arm和参数设置的部分代码，与之前版本相同)
        limit = 0.5
        if 'gibson' in task_name or 'gibson_3smalltree' in task_name: margin = limit / 12.0
        else: margin = limit / 5.0
        offset = margin / 10.0

        input_file = os.path.join(out_path, os.path.splitext(os.path.basename(path))[0] + '_scaled.off')
        
        if task_name == 'arm':
            # ... arm逻辑 ...
            pass
        else:
            print(f"--- Running Budget-Allocated ADAPTIVE Sampling for scene: {out_path} ---")
            mesh = o3d.io.read_triangle_mesh(input_file)
            if not mesh.has_triangles():
                print(f"Error: Mesh file {input_file} is empty or invalid.")
                return

            v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

            start = timer()
            sampled_points, speed = point_rand_sample_bound_points(
                numsamples, dim, v, f, offset, margin, mesh
            )
            end = timer()
            print(f"Adaptive sampling finished in {end - start:.2f} seconds.")

        if 'sampled_points' in locals() and sampled_points.shape[0] > 0:
            np.save(os.path.join(out_path, 'sampled_points.npy'), sampled_points)
            np.save(os.path.join(out_path, 'speed.npy'), speed)
            print(f"Successfully saved .npy files for scene {out_path}")

    except Exception as err:
        print(f'处理 {path} 时发生严重错误: {traceback.format_exc()}')