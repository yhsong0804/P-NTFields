import sys
sys.path.append('.')
from models import model_res_sigmoid as md
from os import path

modelPath = './Experiments/UR5_hang_copy'         
dataPath = './datasets/arm_hang_copy'

model    = md.Model(modelPath, dataPath, 6, device='cuda')

model.train()