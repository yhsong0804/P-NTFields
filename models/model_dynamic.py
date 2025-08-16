import torch
import torch.nn as nn
import numpy as np
import time
from models.model_res_sigmoid_multi import NN, Model, FastTensorDataLoader
from models.model_res_sigmoid_multi import sigmoid, Sigmoid, DSigmoid, sigmoid_out, Sigmoid_out, DSigmoid_out, DDSigmoid_out

class TemporalAttention(nn.Module):
    """时序注意力模块，用于聚合多时刻信息"""
    def __init__(self, feature_dim=128, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True, dropout=0.1)
        
        # 时间位置编码
        self.time_embedding = nn.Linear(1, feature_dim)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, temporal_features, time_indices):
        """
        Args:
            temporal_features: [batch, num_timesteps, feature_dim]
            time_indices: [batch, num_timesteps, 1] 时间索引 [0, 1/3, 2/3, 1]
        """
        # 添加时间位置编码
        time_emb = self.time_embedding(time_indices)  # [batch, timesteps, feature_dim]
        features_with_time = temporal_features + time_emb
        
        # 多头自注意力
        attended_features, attention_weights = self.multihead_attn(
            features_with_time, features_with_time, features_with_time)
        
        # 残差连接和层归一化
        features = self.layer_norm1(temporal_features + attended_features)
        
        # 前馈网络
        ffn_output = self.ffn(features)
        features = self.layer_norm2(features + ffn_output)
        
        return features, attention_weights

class DynamicNN(NN):
    """支持动态障碍物的神经网络"""
    
    def __init__(self, device, dim, num_timesteps=4):
        super().__init__(device, dim)
        self.num_timesteps = num_timesteps
        self.device = device  # 确保设备属性存在
        
        # 时序注意力模块
        self.temporal_attention = TemporalAttention(128, 4)
        
        # 障碍物运动编码器
        self.motion_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 3D运动向量 -> 64维特征
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 动态特征融合层
        self.dynamic_fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),  # 时序特征 + 运动特征
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 修改encoder第一层以支持动态输入
        # 原来: 2*h_size = 256, 现在增加运动信息维度
        self.encoder[0] = nn.Linear(2*128 + 64, 128)

    def encode_motion_context(self, motion_vectors):
        """编码障碍物运动上下文信息"""
        if motion_vectors.dim() == 2:
            # 单个运动向量，扩展为batch
            motion_vectors = motion_vectors.unsqueeze(0)
        
        # 对每个运动向量进行编码
        motion_features = self.motion_encoder(motion_vectors)  # [batch, 64]
        return motion_features

    def forward_dynamic_sequence(self, coords_sequence, B, motion_info):
        """
        处理动态时序输入
        
        Args:
            coords_sequence: [batch, num_timesteps, num_points, 2*dim]
            B: [3, 128] Fourier特征矩阵  
            motion_info: 障碍物运动信息字典
        """
        batch_size, num_timesteps, num_points, coord_dim = coords_sequence.shape
        
        # 处理运动信息
        if not motion_info['is_static']:
            # 计算平均运动向量作为全局运动特征
            motion_vectors = motion_info['motion_vectors']  # [timesteps-1, 3]
            avg_motion = torch.mean(motion_vectors, dim=0, keepdim=True)  # [1, 3]
            motion_features = self.encode_motion_context(avg_motion)  # [1, 64]
            motion_features = motion_features.expand(batch_size, num_points, -1)  # [batch, points, 64]
        else:
            # 静态场景，使用零运动特征
            motion_features = torch.zeros(batch_size, num_points, 64, device=self.device)
        
        # 处理每个时刻的空间特征
        temporal_features = []
        for t in range(num_timesteps):
            coords_t = coords_sequence[:, t]  # [batch, num_points, 2*dim]
            
            # Fourier编码
            coords_t_expanded = coords_t.unsqueeze(2)  # [batch, points, 1, 2*dim]
            x = self.input_mapping(coords_t_expanded, B)  # [batch, points, 1, 256]
            x = x.squeeze(2)  # [batch, points, 256]
            
            # 融合运动信息
            x_with_motion = torch.cat([x, motion_features], dim=-1)  # [batch, points, 320]
            
            # 通过encoder处理
            x = self.act(self.encoder[0](x_with_motion))
            for ii in range(1, self.nl1):
                x_tmp = x
                x = self.act(self.encoder[ii](x))
                x = self.act(self.encoder1[ii](x) + x_tmp)
            
            x = self.encoder[-1](x)  # [batch, points, 128]
            temporal_features.append(x)
        
        # 时序注意力聚合
        temporal_features = torch.stack(temporal_features, dim=1)  # [batch, timesteps, points, 128]
        
        # 为每个点独立应用时序注意力
        attended_features = []
        time_indices = torch.linspace(0, 1, num_timesteps, device=self.device)
        time_indices = time_indices.unsqueeze(0).unsqueeze(-1)  # [1, timesteps, 1]
        time_indices = time_indices.expand(batch_size, -1, -1)   # [batch, timesteps, 1]
        
        for p in range(num_points):
            point_temporal_features = temporal_features[:, :, p, :]  # [batch, timesteps, 128]
            attended_point_features, _ = self.temporal_attention(
                point_temporal_features, time_indices)
            
            # 取时序聚合结果（可以用最后时刻或平均）
            final_point_features = attended_point_features[:, -1, :]  # [batch, 128] 使用最后时刻
            attended_features.append(final_point_features)
        
        attended_features = torch.stack(attended_features, dim=1)  # [batch, points, 128]
        
        # 通过generator处理
        x = attended_features
        x0 = x  # [batch, points, 128]
        x1 = x  # 复制用于双路径处理
        
        xx = torch.cat((x0, x1), dim=2)  # [batch, points, 256]
        x_0 = torch.logsumexp(self.scale * xx, 2) / self.scale
        x_1 = -torch.logsumexp(-self.scale * xx, 2) / self.scale
        x = torch.cat((x_0, x_1), 2)  # [batch, points, 256]
        
        # Generator层
        for ii in range(self.nl2):
            x_tmp = x
            x = self.act(self.generator[ii](x))
            x = self.act(self.generator1[ii](x) + x_tmp)
        
        y = self.generator[-2](x)
        x = self.act(y)
        y = self.generator[-1](x)
        x = self.actout(y)
        
        return x

    def compute_dynamic_velocity_field(self, coords_sequence, B, motion_info):
        """计算动态速度场，考虑障碍物运动产生的圆弧效应"""
        
        # 获取时间场
        tau = self.forward_dynamic_sequence(coords_sequence, B, motion_info)
        
        # 计算最后时刻的梯度（作为当前预测的速度场）
        coords_current = coords_sequence[:, -1]  # [batch, points, 2*dim]
        coords_current = coords_current.clone().detach().requires_grad_(True)
        
        # 使用标准方法计算梯度
        tau_current, dtau, coords = self.out_grad(coords_current, B)
        
        # 如果存在障碍物运动，添加动态修正
        if not motion_info['is_static'] and motion_info['motion_vectors'].numel() > 0:
            # 计算动态修正项，使速度场在障碍物附近呈现圆弧状
            dynamic_correction = self.compute_arc_correction(
                coords_current, dtau, motion_info)
            dtau = dtau + 0.1 * dynamic_correction  # 小幅修正
        
        return tau_current, dtau, coords

    def compute_arc_correction(self, coords, velocity_grad, motion_info):
        """计算圆弧状修正项"""
        batch_size, num_points, coord_dim = coords.shape
        
        # 获取障碍物运动方向
        motion_vectors = motion_info['motion_vectors']  # [timesteps-1, 3]
        avg_motion_dir = torch.mean(motion_vectors, dim=0)  # [3]
        avg_motion_dir = avg_motion_dir / (torch.norm(avg_motion_dir) + 1e-8)  # 归一化
        
        # 扩展到所有点
        motion_dir_expanded = avg_motion_dir.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_points, 3)  # [batch, points, 3]
        
        # 计算与障碍物运动方向的相对角度
        current_velocity = velocity_grad[:, :, self.dim:]  # 终点方向的梯度
        
        # 计算圆弧修正：在垂直于运动方向的平面内添加旋转分量
        # 这会使轨迹呈现圆弧状而不是直线避障
        perpendicular_dir = self.compute_perpendicular_direction(
            current_velocity, motion_dir_expanded)
        
        # 圆弧强度依赖于到障碍物的距离（越近圆弧效应越强）
        obstacle_distance = self.estimate_obstacle_distance(coords, motion_info)
        arc_strength = torch.exp(-obstacle_distance * 5.0)  # 距离衰减
        
        # 应用圆弧修正
        arc_correction = arc_strength.unsqueeze(-1) * perpendicular_dir * 0.1
        
        return torch.cat([arc_correction, arc_correction], dim=-1)  # 起点和终点都修正

    def compute_perpendicular_direction(self, velocity, motion_dir):
        """计算垂直方向，用于圆弧效应"""
        # 简化版本：计算速度向量与运动方向的外积
        cross_product = torch.cross(velocity, motion_dir, dim=-1)
        cross_product = cross_product / (torch.norm(cross_product, dim=-1, keepdim=True) + 1e-8)
        return cross_product

    def estimate_obstacle_distance(self, coords, motion_info):
        """估算到障碍物的距离"""
        # 简化版本：使用障碍物位置计算距离
        if motion_info['is_static']:
            return torch.ones(coords.shape[0], coords.shape[1], device=coords.device)
        
        obstacle_pos = motion_info['obstacle_positions'][-1]  # 最新位置
        obstacle_pos = obstacle_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        
        # 计算到障碍物中心的距离
        coords_pos = coords[:, :, self.dim:]  # 终点位置
        distances = torch.norm(coords_pos - obstacle_pos, dim=-1)
        
        return distances

class DynamicModel(Model):
    """支持动态障碍物的模型类"""
    
    def __init__(self, ModelPath, DataPath, dim, length, device='cpu', num_timesteps=4):
        super().__init__(ModelPath, DataPath, dim, length, device)
        self.num_timesteps = num_timesteps
        
        # 更新训练参数
        self.Params['Training']['Dynamic Timesteps'] = num_timesteps
        self.Params['Training']['Temporal Consistency Weight'] = 0.1
        self.Params['Training']['Arc Constraint Weight'] = 0.05

    def init_dynamic_network(self):
        """初始化动态网络"""
        self.network = DynamicNN(self.Params['Device'], self.dim, self.num_timesteps)
        self.network.apply(self.network.init_weights)
        self.network.to(self.Params['Device'])

    def Loss_Dynamic(self, coords_sequence, Yobs_sequence, B, motion_info, beta, gamma):
        """动态损失函数"""
        
        # 计算每个时刻的静态损失
        static_loss = 0
        temporal_consistency_loss = 0
        
        prev_tau = None
        prev_dtau = None
        
        for t in range(self.num_timesteps):
            coords_t = coords_sequence[:, t]
            Yobs_t = Yobs_sequence[:, t]
            
            # 计算当前时刻的损失
            loss_t, loss_n_t, diff_t = self.Loss_Simple(coords_t, Yobs_t, B, beta, gamma)
            static_loss += loss_t
            
            # 计算时序一致性损失
            if t > 0 and prev_tau is not None:
                tau_t, dtau_t, _ = self.network.out_grad(coords_t, B)
                temporal_consistency_loss += torch.mean((dtau_t - prev_dtau) ** 2)
                prev_tau, prev_dtau = tau_t, dtau_t
            else:
                prev_tau, prev_dtau, _ = self.network.out_grad(coords_t, B)
        
        # 动态轨迹约束损失
        arc_constraint_loss = self.compute_arc_constraint_loss(
            coords_sequence, B, motion_info)
        
        # 总损失
        total_loss = (static_loss + 
                     self.Params['Training']['Temporal Consistency Weight'] * temporal_consistency_loss +
                     self.Params['Training']['Arc Constraint Weight'] * arc_constraint_loss)
        
        return total_loss, static_loss, temporal_consistency_loss, arc_constraint_loss

    def compute_arc_constraint_loss(self, coords_sequence, B, motion_info):
        """计算圆弧约束损失，鼓励在动态障碍物附近形成圆弧轨迹"""
        if motion_info['is_static']:
            return torch.tensor(0.0, device=self.Params['Device'])
        
        # 获取最后时刻的速度场
        coords_current = coords_sequence[:, -1]
        tau, dtau, _ = self.network.out_grad(coords_current, B)
        
        # 计算期望的圆弧方向
        motion_vectors = motion_info['motion_vectors']
        avg_motion = torch.mean(motion_vectors, dim=0)
        
        # 在障碍物附近，速度场应该偏向圆弧方向而不是直线避障
        obstacle_regions = self.identify_obstacle_regions(coords_current, motion_info)
        
        # 计算实际速度方向与期望圆弧方向的差异
        expected_arc_direction = self.compute_expected_arc_direction(
            coords_current, avg_motion, motion_info)
        
        actual_velocity = dtau[:, :, self.dim:]  # 终点方向梯度
        
        # 在障碍物区域内应用圆弧约束
        arc_loss = obstacle_regions * torch.mean(
            (actual_velocity - expected_arc_direction) ** 2, dim=-1)
        
        return torch.mean(arc_loss)

    def identify_obstacle_regions(self, coords, motion_info):
        """识别障碍物影响区域"""
        obstacle_pos = motion_info['obstacle_positions'][-1]  # [3]
        obstacle_pos = obstacle_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        
        coords_pos = coords[:, :, self.dim:]  # [batch, points, 3]
        distances = torch.norm(coords_pos - obstacle_pos, dim=-1)
        
        # 在障碍物附近的区域（距离小于阈值）
        obstacle_influence = torch.exp(-distances * 2.0)  # 距离衰减
        return obstacle_influence

    def compute_expected_arc_direction(self, coords, motion_vector, motion_info):
        """计算期望的圆弧方向"""
        batch_size, num_points, coord_dim = coords.shape
        
        # 简化：期望方向是运动方向的垂直分量
        motion_dir = motion_vector / (torch.norm(motion_vector) + 1e-8)
        motion_dir = motion_dir.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_points, 3)
        
        # 生成垂直方向作为圆弧切线方向
        perpendicular = torch.cross(motion_dir, 
                                  torch.tensor([0, 0, 1], device=motion_dir.device).expand_as(motion_dir), 
                                  dim=-1)
        perpendicular = perpendicular / (torch.norm(perpendicular, dim=-1, keepdim=True) + 1e-8)
        
        return perpendicular