#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from models.model_res_sigmoid_multi import Model

DEVICE      = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATA_PATH   = './datasets/gibson/'
MODEL_PATH  = './Experiments/Gib_multi'
WEIGHT_FILE = os.path.join(MODEL_PATH, 'Model_Epoch_10000_ValLoss_1.221157e-01.pt')

# 1. 加载模型
model = Model(MODEL_PATH, DATA_PATH, 3, 2, device=DEVICE)
model.load(WEIGHT_FILE)
model.network.eval()

# 2. 把 B.npy 也赋给 model.B，这样 Speed/Tau/TravelTimes 内部都能用上
B_path   = os.path.join(DATA_PATH, '0', 'B.npy')
B        = np.load(B_path)
B_tensor = Variable(torch.Tensor(B)).to(DEVICE)
model.B  = B_tensor

# 3. 构造一个 [-0.5,0.5]^2 的 80×80 网格
LIMIT   = 0.5
SPACING = LIMIT / 40.0
x_lin = np.arange(-LIMIT, LIMIT, SPACING)
y_lin = np.arange(-LIMIT, LIMIT, SPACING)
X, Y  = np.meshgrid(x_lin, y_lin)
XY_flat = np.stack([X.flatten(), Y.flatten()], axis=-1)

print(f"Grid shape    : X {X.shape}, Y {Y.shape}")
print(f"Flatten shape : XY_flat {XY_flat.shape}")

def compute_velocity(model, XY_flat, start_xy, device):
    N = XY_flat.shape[0]
    XP_temp = np.zeros((N, 4), dtype=np.float32)
    XP_temp[:, 0] = start_xy[0]
    XP_temp[:, 1] = start_xy[1]
    XP_temp[:, 2] = XY_flat[:, 0]
    XP_temp[:, 3] = XY_flat[:, 1]
    Z_flat = np.zeros((N,1), dtype=np.float32)
    XP_xyz = np.concatenate([
        XP_temp[:,0:2], Z_flat,
        XP_temp[:,2:4], Z_flat
    ], axis=1)   # 形状 (N,6)

    XP_tensor = torch.from_numpy(XP_xyz).to(device)
    with torch.no_grad():
        V_t = model.Speed(XP_tensor)   # 不传 B_tensor，因为 model.B 已经就位
    return V_t.cpu().numpy().reshape(X.shape)

def compute_tau(model, XY_flat, start_xy, device):
    N = XY_flat.shape[0]
    XP_temp = np.zeros((N, 4), dtype=np.float32)
    XP_temp[:, 0] = start_xy[0]
    XP_temp[:, 1] = start_xy[1]
    XP_temp[:, 2] = XY_flat[:, 0]
    XP_temp[:, 3] = XY_flat[:, 1]
    Z_flat = np.zeros((N,1), dtype=np.float32)
    XP_xyz = np.concatenate([
        XP_temp[:,0:2], Z_flat,
        XP_temp[:,2:4], Z_flat
    ], axis=1)

    XP_tensor = torch.from_numpy(XP_xyz).to(device)
    with torch.no_grad():
        tau_t = model.Tau(XP_tensor)
    return tau_t.cpu().numpy().reshape(X.shape)

def compute_tt(model, XY_flat, start_xy, device):
    N = XY_flat.shape[0]
    XP_temp = np.zeros((N, 4), dtype=np.float32)
    XP_temp[:, 0] = start_xy[0]
    XP_temp[:, 1] = start_xy[1]
    XP_temp[:, 2] = XY_flat[:, 0]
    XP_temp[:, 3] = XY_flat[:, 1]
    Z_flat = np.zeros((N,1), dtype=np.float32)
    XP_xyz = np.concatenate([
        XP_temp[:,0:2], Z_flat,
        XP_temp[:,2:4], Z_flat
    ], axis=1)

    XP_tensor = torch.from_numpy(XP_xyz).to(device)
    with torch.no_grad():
        tt_t = model.TravelTimes(XP_tensor)
    return tt_t.cpu().numpy().reshape(X.shape)

# 设置两个起点
start1 = (-0.25, -0.25)
start2 = ( 0.20, -0.10)

# 计算两个速度场并合并
V1 = compute_velocity(model, XY_flat, start1, DEVICE)
V2 = compute_velocity(model, XY_flat, start2, DEVICE)
V_sum  = V1 + V2
V_norm = V_sum / 2.0

# 计算两个时间场并合并
TAU1 = compute_tau(model, XY_flat, start1, DEVICE)
TAU2 = compute_tau(model, XY_flat, start2, DEVICE)
TAU_sum  = TAU1 + TAU2
TAU_norm = TAU_sum / 2.0

# 计算两组的 TravelTime（只用来画等高线）
TT1 = compute_tt(model, XY_flat, start1, DEVICE)
TT2 = compute_tt(model, XY_flat, start2, DEVICE)

os.makedirs('output', exist_ok=True)

# —— 画合并后的速度场 + 双等高线 —— 
out_vel = 'output/vel_sum_with_contours.jpg'
plt.figure(figsize=(6,6))
ax = plt.subplot(111)

# 这里不显式写 cmap，让 matplotlib 用默认的 viridis
quad = ax.pcolormesh(X, Y, V_norm, vmin=0, vmax=1)    # *** 默认 viridis ***
ax.contour(X, Y, TT1, np.arange(0,3,0.05), cmap='bone', linewidths=0.5) 
ax.contour(X, Y, TT2, np.arange(0,3,0.05), colors='black', linewidths=0.5)

plt.colorbar(quad, ax=ax, pad=0.1, label='Combined Predicted Velocity')
plt.title('Combined Velocity Field (start1 + start2) with Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(out_vel, dpi=200, bbox_inches='tight')
plt.close()
print("Saved velocity+contours:", out_vel)

# —— 画合并后的 TAU 场 + 双等高线 —— 
out_tau = 'output/tau_sum_with_contours.jpg'
plt.figure(figsize=(6,6))
ax2 = plt.subplot(111)

quad2 = ax2.pcolormesh(X, Y, TAU_norm, vmin=0, vmax=1)  # *** 默认 viridis ***
ax2.contour(X, Y, TT1, np.arange(0,3,0.05), cmap='bone', linewidths=0.5)
ax2.contour(X, Y, TT2, np.arange(0,3,0.05), colors='black', linewidths=0.5)

plt.colorbar(quad2, ax=ax2, pad=0.1, label='Combined Predicted Time-to-Go')
plt.title('Combined Tau Field (start1 + start2) with Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(out_tau, dpi=200, bbox_inches='tight')
plt.close()
print("Saved tau+contours:", out_tau)
