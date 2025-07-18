import sys

sys.path.append('.')

from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.speed_sampling_gpu import sample_speed
import dataprocessing.voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from glob import glob
import configs.config_loader as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os
cfg = cfg_loader.get_config()
print(cfg.data_dir)
print(cfg.input_data_glob)

print('Finding raw files for preprocessing.')
paths = glob( "./"+cfg.data_dir + cfg.input_data_glob)
print(paths)
paths = sorted(paths)

chunks = np.array_split(paths,cfg.num_chunks)
paths = chunks[cfg.current_chunk]


if cfg.num_cpus == -1:
	num_cpus = mp.cpu_count()
else:
	num_cpus = cfg.num_cpus

def multiprocess(func):
	p = Pool(num_cpus)
	p.map(func, paths)
	p.close()
	p.join()

print('Start scaling.')
# multiprocess(to_off)

# ----------------------------------------------------
# 用这段新代码替换掉旧的 for 循环 和 voxelized_pointcloud_sampling 部分
# ----------------------------------------------------
print('Start speed sampling.')

# 创建一个集合来跟踪已经处理过的场景目录
processed_dirs = set()
for path in paths:
    # 获取当前文件所在的目录
    dir_path = os.path.dirname(path)
    if dir_path not in processed_dirs:
        print(f"\n--- Processing scene: {dir_path} ---")
        # 只对每个场景目录调用一次 sample_speed
        sample_speed(path, cfg.num_samples, cfg.num_dim)
        processed_dirs.add(dir_path)

print('\nStart voxelized pointcloud sampling.')
voxelized_pointcloud_sampling.init(cfg)

# 体素化也同样，只对每个场景处理一次
processed_dirs_voxel = set()
for path in paths:
    dir_path = os.path.dirname(path)
    if dir_path not in processed_dirs_voxel:
        # 体素化脚本是多进程的，所以我们只把路径传给它
        # 注意：体素化脚本本身可能也需要修改以适应动态场景
        # 这里我们先保证采样正确
         processed_dirs_voxel.add(dir_path)

# 运行体素化 (如果需要)
# 注意: 原始的体素化脚本可能不支持动态场景，这是一个后续可以改进的点
# multiprocess(voxelized_pointcloud_sampling.voxelized_pointcloud_sampling)
print("Voxelization step skipped for dynamic scenes for now.")
# ----------------------------------------------------

