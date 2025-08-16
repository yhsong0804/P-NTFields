#!/usr/bin/env python3
# gpt/voxel_test.py

import matplotlib
matplotlib.use('Agg')   # 切换到无 GUI 后端
import matplotlib.pyplot as plt


import sys, os
# 1) 把项目根目录加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 2) 强制指定使用 configs/gibson.txt
#    注意：后续 get_config() 会从 sys.argv 里读取 --config
sys.argv.extend(['--config', 'configs/gibson.txt'])

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# 3) 导入预处理模块
from dataprocessing.voxelized_pointcloud_sampling import init, voxelized_pointcloud_sampling
import configs.config_loader as cfg_loader

def main():
    # 4) 加载配置并初始化全局 kdtree 和 grid_points
    cfg = cfg_loader.get_config()
    init(cfg)

    # 5) 指定我们要处理的 .off 文件
    off_path = 'datasets/gibson/0/mesh_z_up_scaled.off'
    # 执行体素化点云采样
    voxelized_pointcloud_sampling(off_path)

    # 6) 读回 .npz 文件
    out_dir = os.path.dirname(off_path)
    npz_path = os.path.join(
        out_dir,
        f'voxelized_point_cloud_{cfg.input_res}res_{cfg.num_points}points.npz'
    )
    print(f"Loading voxel data from {npz_path}")
    data = np.load(npz_path)

    # 7) 还原 occupancy 体积
    comp_flat = data['compressed_occupancies']        # bit-packed
    res       = int(data['res'])                    # 分辨率
    occ_flat  = np.unpackbits(comp_flat)[:res**3]   # 展平成 res^3 长度
    occ_vol   = occ_flat.reshape((res, res, res))   # 还原成 (res,res,res)

    # 8) 画 Z 轴中间层的切片
    mid = res // 2
    slice2d = occ_vol[:, :, mid]

    plt.figure(figsize=(6,6))
    plt.imshow(slice2d.T, origin='lower', cmap='gray')
    plt.title(f'Voxel occupancy slice at z-index={mid}')
    plt.xlabel('Voxel X index'); plt.ylabel('Voxel Y index')
    plt.colorbar(label='Occupied=1 / Free=0')
    plt.tight_layout()
    plt.savefig('gpt/voxel_test.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved figure to gpt/voxel_test.png")


    # 9) 可选：也可用 Open3D 可视化表面点云
    #    Uncomment if you want to see the surface points too.
    # pts = data['point_cloud']
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts)
    # o3d.visualization.draw_geometries([pcd], window_name='Surface Point Cloud')

if __name__ == "__main__":
    main()




