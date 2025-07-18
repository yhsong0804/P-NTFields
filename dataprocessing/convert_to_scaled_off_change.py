# dataprocessing/convert_to_scaled_off.py

import os
import sys
import traceback
import logging
import igl
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def to_off(path):
    """
    将原始 mesh 转成居中归一化的 OFF，并在同目录下写入 mesh_normalization.npz
    """
    file_path = os.path.dirname(path)
    # 例如 datasets/gibson_3smalltree/0/mesh_z_up.obj
    # 则 data_type = 'gibson_3smalltree'
    data_type = file_path.split(os.sep)[2]
    file_name  = os.path.splitext(os.path.basename(path))[0]
    output_off = os.path.join(file_path, file_name + '_scaled.off')

    if os.path.exists(output_off):
        print(f'Exists: {output_off}')
    try:
        # 1) 读 mesh
        v, f = igl.read_triangle_mesh(path)

        # 2) 计算轴对齐包围盒
        bb_max  = v.max(axis=0, keepdims=True)   # shape (1,3)
        bb_min  = v.min(axis=0, keepdims=True)   # shape (1,3)
        centers = (bb_max + bb_min) / 2.0        # 中心
        scales  = (bb_max - bb_min)             # 尺度

        # 3) 针对不同类型数据做不同处理
        if data_type == 'c3d':
            v = v / 40.0
        elif data_type == 'arm':
            # arm 已经在 URDF 中定义，不做归一化
            pass
        else:
            # 其它场景：居中 & 归一化到单位立方体
            v = (v - centers) / scales

        # 4) 写出 OFF
        igl.write_triangle_mesh(output_off, v, f)
        print(f'Finished: {path}')

        # 5) **关键**：在同一个目录下写入归一化参数
        norm_file = os.path.join(file_path, 'mesh_normalization.npz')
        np.savez(norm_file,
                 center=centers.flatten(),  # (3,)
                 scale = scales.flatten())  # (3,)
        print(f'Wrote normalization params: {norm_file}')

    except Exception:
        print(f'Error with {path}: {traceback.format_exc()}')

if __name__ == '__main__':
    import glob
    # 你也可以添加脚本式调用，遍历所有 .obj/.off
    for obj_path in glob.glob('datasets/*/*/*.obj'):
        to_off(obj_path)
