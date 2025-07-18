#data_multi.py修改为dynamic版本

# ==============================================================================
# 1. 用这个【新版本】的 Database 类，完整地替换掉旧的 Database 类
# ==============================================================================

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad
import os # 确保导入了 os 模块

class Database(torch.utils.data.Dataset):
    def __init__(self, path, device, len):
        # 创建 идентичные пары
        self.device = device
        self.path = path
        self.len = len

    def __getitem__(self, index):
        # 构造当前场景的路径
        scene_path = os.path.join(self.path, str(index))
        
        # 加载所有必需的数据文件
        points = np.load(os.path.join(scene_path, 'sampled_points.npy'))
        speed = np.load(os.path.join(scene_path, 'speed.npy'))
        B = np.load(os.path.join(scene_path, 'B.npy'))
        
        # 【新逻辑】: 加载我们新增的时间戳文件
        timestamps_path = os.path.join(scene_path, 'timestamps.npy')
        try:
            timestamps = np.load(timestamps_path)
            print(f"成功为场景 {index} 加载 timestamps.npy")
        except FileNotFoundError:
            # 如果找不到时间戳文件，就创建一个默认值，以兼容旧的静态场景
            print(f"警告: 在场景 {index} 中找不到 timestamps.npy，将使用默认值 0。")
            timestamps = np.zeros((points.shape[0], 1), dtype=np.int64)

        # 转换为Tensor
        points = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        B = Variable(Tensor(B))
        timestamps = Variable(Tensor(timestamps)).squeeze().long() # 确保是1维的长整型

        # 将 points 和 speed 合并成一个 data 张量
        data = torch.cat((points,speed),dim=1)
        
        # 【变化】: 现在返回一个包含时间戳的元组
        return data, B, timestamps

    def __len__(self):
        return self.len