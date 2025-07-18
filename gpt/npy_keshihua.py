import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 1. 找到所有error_points文件
files = glob.glob('./Experiments/Gib_multi/error_points_epoch_*.npy')
files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按epoch排序

epochs = []
mean_errors = []

for f in files:
    epoch = int(f.split('_')[-1].split('.')[0])
    error_arr = np.load(f)
    mean_error = error_arr.mean()  # 可以也统计max/min/std
    epochs.append(epoch)
    mean_errors.append(mean_error)
plt.figure(figsize=(8,5))
plt.plot(epochs, mean_errors, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Error')
plt.title('Mean Error vs Epoch')
plt.grid(True)
plt.tight_layout()
plt.savefig('error_curve.png', dpi=200)   # 保存为png
plt.show()
