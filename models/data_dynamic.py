import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad
import os

class Database(torch.utils.data.Dataset):
    def __init__(self, path, device, len):
        # Creating identical pairs
        self.device = device
        self.path = path
        self.len = len

    def __getitem__(self, index):
        path = self.path + str(index)
        points = np.load('{}/sampled_points.npy'.format(path)).astype(np.float16)
        speed = np.load('{}/speed.npy'.format(path))
        B = np.load('{}/B.npy'.format(path))

        points = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        B = Variable(Tensor(B))
        
        data = torch.cat((points,speed),dim=1)
        return data, B, index
    
    def __len__(self):
        return self.len

class DynamicDatabase(torch.utils.data.Dataset):
    """
    动态障碍物数据加载器
    加载时序mesh对应的采样点和速度场数据
    """
    def __init__(self, path, device, len, num_timesteps=4):
        self.device = device
        self.path = path
        self.len = len
        self.num_timesteps = num_timesteps

    def __getitem__(self, index):
        """
        返回一个完整的时序序列数据
        
        Returns:
            data_sequence: [num_timesteps, num_points, 8] (6维坐标+2维速度)
            B: [3, 128] 共享的Fourier特征矩阵
            motion_info: 障碍物运动信息
            index: 数据索引
        """
        base_path = self.path + str(index)
        
        # 检查是否存在时序数据文件
        if not self._check_timestep_files(base_path):
            # 如果没有时序文件，使用静态数据进行填充
            return self._load_static_fallback(base_path, index)
        
        # 加载时序数据
        data_sequence = []
        
        for t in range(self.num_timesteps):
            # 加载第t时刻的数据
            points_t = np.load(f'{base_path}/sampled_points_t{t}.npy').astype(np.float16)
            speed_t = np.load(f'{base_path}/speed_t{t}.npy')
            
            points_t = Variable(Tensor(points_t))
            speed_t = Variable(Tensor(speed_t))
            
            # 拼接坐标和速度
            data_t = torch.cat((points_t, speed_t), dim=1)
            data_sequence.append(data_t)
        
        # 加载共享的Fourier特征
        B = np.load(f'{base_path}/B.npy')
        B = Variable(Tensor(B))
        
        # 加载障碍物运动信息
        motion_info = self._load_motion_info(base_path)
        
        # 转换为tensor序列
        data_sequence = torch.stack(data_sequence, dim=0)  # [num_timesteps, num_points, 8]
        
        return data_sequence, B, motion_info, index

    def _check_timestep_files(self, base_path):
        """检查是否存在所有时序数据文件"""
        for t in range(self.num_timesteps):
            points_file = f'{base_path}/sampled_points_t{t}.npy'
            speed_file = f'{base_path}/speed_t{t}.npy'
            if not (os.path.exists(points_file) and os.path.exists(speed_file)):
                return False
        return True

    def _load_static_fallback(self, base_path, index):
        """当没有时序数据时，使用静态数据作为后备方案"""
        print(f"Warning: No timestep data found for {base_path}, using static fallback")
        
        # 加载静态数据
        points = np.load(f'{base_path}/sampled_points.npy').astype(np.float16)
        speed = np.load(f'{base_path}/speed.npy')
        B = np.load(f'{base_path}/B.npy')
        
        points = Variable(Tensor(points))
        speed = Variable(Tensor(speed))
        B = Variable(Tensor(B))
        
        data = torch.cat((points, speed), dim=1)
        
        # 复制静态数据到所有时刻
        data_sequence = data.unsqueeze(0).repeat(self.num_timesteps, 1, 1)
        
        # 创建空的运动信息
        motion_info = {
            'obstacle_positions': torch.zeros(self.num_timesteps, 3),
            'motion_vectors': torch.zeros(self.num_timesteps-1, 3),
            'is_static': True
        }
        
        return data_sequence, B, motion_info, index

    def _load_motion_info(self, base_path):
        """加载障碍物运动信息"""
        motion_file = f'{base_path}/motion_info.npy'
        
        if os.path.exists(motion_file):
            motion_data = np.load(motion_file, allow_pickle=True).item()
            motion_info = {
                'obstacle_positions': torch.tensor(motion_data['positions']),
                'motion_vectors': torch.tensor(motion_data['vectors']),
                'is_static': False
            }
        else:
            # 如果没有运动信息文件，尝试从mesh文件中提取
            try:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from dataprocessing.speed_sampling_dynamic import extract_obstacle_motion
                positions, vectors = extract_obstacle_motion(base_path, self.num_timesteps)
                motion_info = {
                    'obstacle_positions': torch.tensor(positions),
                    'motion_vectors': torch.tensor(vectors),
                    'is_static': False
                }
                # 保存提取的运动信息
                np.save(motion_file, {
                    'positions': positions, 
                    'vectors': vectors
                })
            except:
                # 如果提取失败，使用默认值
                motion_info = {
                    'obstacle_positions': torch.zeros(self.num_timesteps, 3),
                    'motion_vectors': torch.zeros(self.num_timesteps-1, 3),
                    'is_static': True
                }
        
        return motion_info

    def __len__(self):
        return self.len

    def get_timestep_data(self, index, timestep):
        """获取特定时刻的数据（用于单时刻训练）"""
        if timestep >= self.num_timesteps:
            raise ValueError(f"Timestep {timestep} exceeds max timesteps {self.num_timesteps}")
        
        base_path = self.path + str(index)
        
        points_t = np.load(f'{base_path}/sampled_points_t{timestep}.npy').astype(np.float16)
        speed_t = np.load(f'{base_path}/speed_t{timestep}.npy')
        B = np.load(f'{base_path}/B.npy')
        
        points_t = Variable(Tensor(points_t))
        speed_t = Variable(Tensor(speed_t))
        B = Variable(Tensor(B))
        
        data_t = torch.cat((points_t, speed_t), dim=1)
        return data_t, B