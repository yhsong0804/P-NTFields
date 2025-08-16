import sys
sys.path.append('.')
from models import model_res_sigmoid_multi as md
from os import path

print("=== 训练脚本：Gibson场景 + 小障碍物优化 ===")

modelPath = './Experiments/Gib_multi_small_objects'
#dataPath = './datasets/gibson/'
dataPath = './datasets/gibson_3smalltree/my_final_scene2/'

print(f"模型保存路径: {modelPath}")
print(f"数据路径: {dataPath}")

# 创建模型输出目录
import os
if not os.path.exists(modelPath):
    os.makedirs(modelPath)
    print(f"创建模型目录: {modelPath}")

model = md.Model(modelPath, dataPath, 3, 2, device='cuda:0')

print("开始训练...")
model.train()

print("训练完成！")