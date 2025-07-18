#!/usr/bin/env python3
# gpt/speed_sample_demo.py
#第二步速度场采样的脚本
import os
import numpy as np
import igl
import matplotlib.pyplot as plt

def sample_and_save_speed(off_file, out_dir,
                          num_samples=20000,
                          dim=2,
                          d_max=0.5/12.0):
    """
    1) 从 off_file（已缩放到 [-0.5,0.5]^3 的网格）随机采 num_samples 个点
    2) 用 igl.signed_distance 计算每个点到三角网格的最短距离
    3) 把距离 / d_max clip 到 [0,1] 得到速度标签
    4) 保存 (points.npy, speeds.npy) 并画个散点图可视化
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. 载入网格
    V, F = igl.read_triangle_mesh(off_file)
    print(f"Loaded mesh: {off_file}, #V={len(V)}, #F={len(F)}")

    # 2. 随机采样点
    pts2d = np.random.rand(num_samples, dim) * 1.0 - 0.5   # 均匀采样 [-0.5,0.5]^2
    pts3d = np.zeros((num_samples, 3))
    pts3d[:, :dim] = pts2d

    # 3. 计算无符号距离 —— 只取返回的第一个值作为距离
    sd = igl.signed_distance(pts3d, V, F)
    # sd 可能是长度 3 也可能是长度 4 的 tuple/list
    dist = np.abs(sd[0])   # 拿第一个元素并取绝对值

    # 4. 线性映射到速度 [0,1]
    speeds = np.clip(dist / d_max, 0.0, 1.0)

    # 5. 保存
    np.save(os.path.join(out_dir, 'sampled_points.npy'), pts2d)
    np.save(os.path.join(out_dir, 'speeds.npy'),        speeds)
    print(f"Saved sampled_points.npy and speeds.npy to {out_dir}")

    # 6. 可视化散点
    plt.figure(figsize=(6,6))
    sc = plt.scatter(pts2d[:,0], pts2d[:,1],
                     c=speeds, cmap='viridis',
                     s=1, vmin=0, vmax=1)
    plt.colorbar(sc, label='Speed S(q)')
    plt.title('Random Speed Samples')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.axis('equal')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'speed_samples.png')
    plt.savefig(fig_path, dpi=200)
    print(f"Saved visualization to {fig_path}")
    plt.show()

if __name__ == "__main__":
    # —— 针对你的环境，修改这两个路径 —— 
    off_file = 'datasets/gibson/0/mesh_z_up_scaled.off'
    out_dir   = 'datasets/gibson/0/'

    sample_and_save_speed(off_file, out_dir,
                          num_samples=20000,
                          dim=2,
                          d_max=0.5/12.0)
