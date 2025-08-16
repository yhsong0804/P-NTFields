import sys

sys.path.append('.')

from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.speed_sampling_gpu_enhanced_small_objects import sample_speed  # 使用增强版采样
import dataprocessing.voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from glob import glob
import configs.config_loader as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os
import igl

print("=== 增强版预处理脚本 - 专门针对小障碍物优化 ===")

cfg = cfg_loader.get_config()
print(f"数据目录: {cfg.data_dir}")
print(f"输入数据模式: {cfg.input_data_glob}")

print('查找需要预处理的原始文件...')
paths = glob( "./"+cfg.data_dir + cfg.input_data_glob)
print(f"找到文件: {paths}")
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

def generate_pc_npy(path):
	"""生成pc.npy文件 - 从pc.obj转换而来"""
	try:
		out_path = os.path.dirname(path)
		pc_obj_file = os.path.join(out_path, 'pc.obj')
		pc_npy_file = os.path.join(out_path, 'pc.npy')
		
		if os.path.exists(pc_npy_file):
			print(f'Exists: {pc_npy_file}')
			return
			
		if os.path.exists(pc_obj_file):
			# 从pc.obj读取点云数据
			v, f = igl.read_triangle_mesh(pc_obj_file)  # f会是空的，因为这是点云
			# 保存为pc.npy
			np.save(pc_npy_file, v)
			print(f'Generated: {pc_npy_file}')
		else:
			print(f'Warning: {pc_obj_file} not found, skipping pc.npy generation')
	except Exception as err:
		print(f'Error generating pc.npy for {path}: {err}')

print('开始模型缩放处理...')
multiprocess(to_off)

print('开始增强版速度场采样 (专门针对小障碍物优化)...')
for path in paths:
	print(f"处理文件: {path}")
	sample_speed(path, cfg.num_samples, cfg.num_dim)

print('开始体素化点云采样...')
voxelized_pointcloud_sampling.init(cfg)
multiprocess(voxelized_pointcloud_sampling.voxelized_pointcloud_sampling)

print('生成pc.npy文件...')
multiprocess(generate_pc_npy)

print("=== 增强版预处理完成 ===")

# 验证生成的文件
print("\n=== 文件生成验证 ===")
for path in paths:
	out_path = os.path.dirname(path)
	required_files = [
		'mesh_z_up_scaled.off',  # scaling步骤
		'sampled_points.npy',    # speed sampling步骤
		'speed.npy',             # speed sampling步骤  
		'B.npy',                 # speed sampling步骤
		'pc.obj',                # voxelized pointcloud步骤
		'pc.npy',                # 新增的pc.npy
		f'voxelized_point_cloud_{cfg.input_res}res_{cfg.num_points}points.npz'  # voxelized pointcloud步骤
	]
	
	print(f"\n检查目录: {out_path}")
	for file_name in required_files:
		file_path = os.path.join(out_path, file_name)
		if os.path.exists(file_path):
			print(f"  ✓ {file_name}")
		else:
			print(f"  ✗ {file_name} - 缺失!")

print("\n=== 预处理完成 ===")