import sys
sys.path.append('.')
from models import model_res_sigmoid as md
from os import path

modelPath = './Experiments/Arm'  
modelPath = './Experiments/UR5'         
dataPath = './datasets/arm/UR5'

model    = md.Model(modelPath, dataPath, 6, device='cuda')

model.train()