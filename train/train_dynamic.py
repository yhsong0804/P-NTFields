# 动态障碍物PINN训练脚本
import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import time

from models.model_dynamic import DynamicModel, DynamicNN
from models.data_dynamic import DynamicDatabase
from dataprocessing.speed_sampling_dynamic import sample_dynamic_speed_sequence

class DynamicTrainer:
    """动态障碍物PINN训练器"""
    
    def __init__(self, model_path, data_path, dim=3, length=2, device='cuda:0', num_timesteps=4):
        self.model_path = model_path
        self.data_path = data_path
        self.dim = dim
        self.length = length
        self.device = device
        self.num_timesteps = num_timesteps
        
        # 创建模型保存目录
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(f"{model_path}/plots", exist_ok=True)
        
        # 初始化模型
        self.model = DynamicModel(model_path, data_path, dim, length, device, num_timesteps)
        self.model.init_dynamic_network()
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(
            self.model.network.parameters(), 
            lr=1e-3, 
            weight_decay=0.1
        )
    
    def prepare_dynamic_dataset(self):
        """准备动态数据集"""
        print("Preparing dynamic dataset...")
        
        # 检查是否存在时序数据，如果不存在则生成
        for scene_idx in range(self.length):
            scene_path = f"{self.data_path}{scene_idx}"
            
            # 检查是否存在时序文件
            timestep_files_exist = all(
                os.path.exists(f"{scene_path}/sampled_points_t{t}.npy") and
                os.path.exists(f"{scene_path}/speed_t{t}.npy")
                for t in range(self.num_timesteps)
            )
            
            if not timestep_files_exist:
                print(f"Generating timestep data for scene {scene_idx}...")
                
                # 检查是否存在时序mesh文件
                mesh_files_exist = all(
                    os.path.exists(f"{scene_path}/mesh_z_up_t{t}.obj")
                    for t in range(self.num_timesteps)
                )
                
                if mesh_files_exist:
                    # 从时序mesh生成采样数据
                    try:
                        sample_dynamic_speed_sequence(
                            scene_path, 50000, self.dim, self.num_timesteps)
                        print(f"Successfully generated data for scene {scene_idx}")
                    except Exception as e:
                        print(f"Error generating data for scene {scene_idx}: {e}")
                        print("Will use static fallback during training")
                else:
                    print(f"No timestep mesh files found for scene {scene_idx}")
                    print("Will use static fallback during training")
        
        # 创建数据加载器
        self.dataset = DynamicDatabase(
            self.data_path, 
            torch.device(self.device), 
            self.length, 
            self.num_timesteps
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # 动态数据通常batch_size=1
            num_workers=1,
            shuffle=True
        )
        
        print(f"Dataset prepared with {len(self.dataset)} scenes")
    
    def train_dynamic(self, num_epochs=5000):
        """动态训练主循环"""
        print(f"Starting dynamic training for {num_epochs} epochs...")
        
        self.prepare_dynamic_dataset()
        
        # 训练参数
        beta = 1.0
        gamma = 0.001
        print_every = 10
        save_every = 100
        plot_every = 100
        
        # 训练循环
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            total_loss = 0
            total_static_loss = 0
            total_temporal_loss = 0
            total_arc_loss = 0
            
            for batch_idx, (data_sequence, B, motion_info, index) in enumerate(self.dataloader):
                # 移动数据到GPU
                data_sequence = data_sequence.to(self.device)  # [batch, timesteps, points, 8]
                B = B.to(self.device)  # [batch, 3, 128]
                
                # 移动motion_info到GPU
                for key in motion_info:
                    if isinstance(motion_info[key], torch.Tensor):
                        motion_info[key] = motion_info[key].to(self.device)
                
                # 分离坐标和速度
                coords_sequence = data_sequence[:, :, :, :2*self.dim]  # [batch, timesteps, points, 6]
                speed_sequence = data_sequence[:, :, :, 2*self.dim:]   # [batch, timesteps, points, 2]
                
                # 确保requires_grad
                coords_sequence.requires_grad_(True)
                
                # 计算动态损失
                if hasattr(self.model, 'Loss_Simple'):
                    # 使用简化损失函数避免维度问题
                    loss, static_loss, temporal_loss = self.model.Loss_Simple(
                        coords_current, speed_current, B[0], beta, gamma
                    )
                    arc_loss = torch.tensor(0.0)
                elif hasattr(self.model, 'Loss_Dynamic'):
                    # 使用新的动态损失函数
                    loss, static_loss, temporal_loss, arc_loss = self.model.Loss_Dynamic(
                        coords_sequence, speed_sequence, B[0], motion_info, beta, gamma
                    )
                else:
                    # 回退到普通损失函数（对最后时刻）
                    coords_current = coords_sequence[:, -1]
                    speed_current = speed_sequence[:, -1]
                    loss, static_loss, _ = self.model.Loss(
                        coords_current, speed_current, B[0], beta, gamma
                    )
                    temporal_loss = torch.tensor(0.0)
                    arc_loss = torch.tensor(0.0)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 累计损失
                total_loss += loss.item()
                total_static_loss += static_loss.item()
                total_temporal_loss += temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss
                total_arc_loss += arc_loss.item() if isinstance(arc_loss, torch.Tensor) else arc_loss
            
            # 平均损失
            avg_loss = total_loss / len(self.dataloader)
            avg_static_loss = total_static_loss / len(self.dataloader)
            avg_temporal_loss = total_temporal_loss / len(self.dataloader)
            avg_arc_loss = total_arc_loss / len(self.dataloader)
            
            # 更新beta（可选）
            beta = 1.0 / max(avg_static_loss, 1e-6)
            
            # 打印进度
            if epoch % print_every == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4e} | "
                      f"Static: {avg_static_loss:.4e} | "
                      f"Temporal: {avg_temporal_loss:.4e} | "
                      f"Arc: {avg_arc_loss:.4e} | "
                      f"Time: {epoch_time:.2f}s")
            
            # 保存模型
            if epoch % save_every == 0 or epoch == num_epochs:
                self.save_model(epoch, avg_loss)
            
            # 生成可视化
            if epoch % plot_every == 0 or epoch == num_epochs:
                self.plot_dynamic_results(epoch, avg_loss)
    
    def save_model(self, epoch, loss):
        """保存模型"""
        save_path = f"{self.model_path}/Dynamic_Model_Epoch_{epoch:05d}_Loss_{loss:.6e}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'num_timesteps': self.num_timesteps
        }, save_path)
        
        print(f"Model saved: {save_path}")
    
    def load_model(self, checkpoint_path):
        """加载预训练模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['loss']
    
    def plot_dynamic_results(self, epoch, loss):
        """可视化动态结果"""
        print(f"Generating plots for epoch {epoch}...")
        
        with torch.no_grad():
            # 创建测试网格
            limit = 0.5
            spacing = limit / 40.0
            X, Y = np.meshgrid(
                np.arange(-limit, limit, spacing),
                np.arange(-limit, limit, spacing)
            )
            
            # 创建测试点
            Xsrc = [-0.25, -0.25, 0]  # 3D起点
            XP = np.zeros((len(X.flatten()), 2 * self.dim))
            XP[:, :self.dim] = Xsrc
            XP[:, self.dim:self.dim+2] = np.column_stack([X.flatten(), Y.flatten()])
            if self.dim == 3:
                XP[:, self.dim+2] = 0  # Z坐标设为0
            
            XP_tensor = torch.tensor(XP, device=self.device, dtype=torch.float32)
            
            # 使用第一个场景的B矩阵
            try:
                B = np.load(f"{self.data_path}0/B.npy")
                B_tensor = torch.tensor(B, device=self.device, dtype=torch.float32)
            except:
                # 如果没有B文件，生成一个默认的
                B_tensor = torch.randn(3, 128, device=self.device) * 0.5
            
            # 计算速度场和时间场
            if hasattr(self.model.network, 'out_grad'):
                tau, dtau, _ = self.model.network.out_grad(XP_tensor, B_tensor)
                speed_field = self.model.Speed(XP_tensor)
                time_field = tau[:, 0]
            else:
                # 简化版本
                tau, _ = self.model.network.out(XP_tensor, B_tensor)
                speed_field = torch.ones_like(tau[:, 0])
                time_field = tau[:, 0]
            
            # 转换为numpy用于绘图
            speed_field = speed_field.cpu().numpy().reshape(X.shape)
            time_field = time_field.cpu().numpy().reshape(X.shape)
            
            # 绘制速度场
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 速度场
            im1 = ax1.pcolormesh(X, Y, speed_field, vmin=0, vmax=1, cmap='viridis')
            ax1.set_title(f'Dynamic Velocity Field (Epoch {epoch})')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            plt.colorbar(im1, ax=ax1, label='Velocity')
            
            # 时间场
            im2 = ax2.pcolormesh(X, Y, time_field, cmap='plasma')
            ax2.set_title(f'Time Field (Epoch {epoch})')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            plt.colorbar(im2, ax=ax2, label='Time')
            
            plt.tight_layout()
            
            # 保存图像
            plot_path = f"{self.model_path}/plots/dynamic_epoch_{epoch:05d}_loss_{loss:.4e}.jpg"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved: {plot_path}")

def main():
    """主函数"""
    # 训练配置
    config = {
        'model_path': './Experiments/Gib_dynamic_multi/',
        'data_path': './datasets/gibson_3smalltree_dynamic/my_final_scene/',
        'dim': 3,
        'length': 2,  # 场景数量
        'device': 'cuda:0',
        'num_timesteps': 4,
        'num_epochs': 3000
    }
    
    # 创建训练器
    trainer = DynamicTrainer(**{k: v for k, v in config.items() if k != 'num_epochs'})
    
    # 开始训练
    trainer.train_dynamic(config['num_epochs'])
    
    print("Dynamic training completed!")

if __name__ == "__main__":
    main()