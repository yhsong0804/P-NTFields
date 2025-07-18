import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import random
import sys
import traceback
import logging
import igl
import numpy as np
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
#隐藏输出的小工具类
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#mesh scaling主函数
def to_off(path): #路径 os（python模块下的路径处理函数）

    file_path = os.path.dirname(path) #文件目录
    data_type = file_path.split('/')[2] #取类型
    #print(data_type)
    file_name = os.path.splitext(os.path.basename(path))[0] #去掉扩展名
    output_file = os.path.join(file_path,file_name + '_scaled.off')#输出名

    if os.path.exists(output_file):
        print('Exists: {}'.format(output_file)) #已存在则提示
        #return
#尝试读mesh并归一化
    try:
        #with HiddenPrints():已经注释
        '''
        input = trimesh.load(path)
        mesh = as_mesh(input)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        mesh.export(output_file)
        '''
        v, f = igl.read_triangle_mesh(path) # igl读mesh      v：顶点坐标 f：面片顶点索引

        bb_max = v.max(axis=0, keepdims=True)#每个坐标轴最大值
        bb_min = v.min(axis=0, keepdims=True)#每个坐标轴最小值
        #print(centers)
        #print(bb_max-bb_min)
        #按类型归一化
        if data_type == 'c3d':
            v/=40
        elif data_type == 'arm':
            v =v
            
        else:#如gibson
            centers = (bb_max+bb_min)/2.0#中心点
            v = v-centers#平移到原点
            v = v/(bb_max-bb_min)#缩放到每边最大为1
            print(centers)
            print((bb_max-bb_min))
#写入off mesh            
        igl.write_triangle_mesh(output_file, v, f) #保存mesh为off

        print('Finished: {}'.format(path))#打印完成
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))#打印错误
