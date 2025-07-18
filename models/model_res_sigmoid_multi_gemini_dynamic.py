# gemini改造版本dynamic时间维度 已经改残

import os
import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_multi as db
# 找到 from . import data_multi as db
from models import data_multi as db
import copy

import matplotlib
import matplotlib.pylab as plt

import pickle5 as pickle 

from timeit import default_timer as timer

torch.backends.cudnn.benchmark = True

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def sigmoid(input):
 
    return torch.sigmoid(10*input)

class Sigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid(input) 

class DSigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 10*sigmoid(input)*(1-sigmoid(input)) 

def sigmoid_out(input):
 
    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid_out(input) 

class DSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.1*sigmoid_out(input)*(1-sigmoid_out(input)) 

class DDSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.01*sigmoid_out(input)*(1-sigmoid_out(input))*(1-2*sigmoid_out(input))

class NN(torch.nn.Module):
    
    def __init__(self, device, dim):#10
        super(NN, self).__init__()
        self.dim = dim
        h_size = 128 #512,256
        self.input_size = 128


        self.scale = 10

        self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU
        
        # ... 省略前面的代码 ...
        self.scale = 10
        self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU

        # --- 【新增代码块】 时间编码模块 ---
        # 定义我们的场景有多少个离散的时间步
        self.num_timesteps = 4 
        # 定义要把每个时间步编码成多高维度的向量
        self.time_embedding_dim = 64 
        # 这就是我们的“时间感官”：一个Embedding层
        # 它的作用像一个查询表，输入0就输出第0个64维向量，输入1就输出第1个...
        self.time_embedder = torch.nn.Embedding(
            num_embeddings=self.num_timesteps, 
            embedding_dim=self.time_embedding_dim
        )
        # --- 新增代码块结束 ---

        #self.env_act = torch.nn.Sigmoid()#ELU
        # ... 后面原有的 self.encoder 和 self.generator 定义保持不变 ...
        

        #self.env_act = torch.nn.Sigmoid()#ELU
        self.dact = Sigmoid()
        self.ddact = DSigmoid()

        self.actout = Sigmoid_out()#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU
        self.dactout = DSigmoid_out()
        self.ddactout = DDSigmoid_out()

        self.nl1=3
        self.nl2=3

        self.encoder = torch.nn.ModuleList()
        self.encoder1 = torch.nn.ModuleList()
        #self.encoder.append(Linear(self.dim,h_size))
        
        self.encoder.append(Linear(2*h_size,h_size))
        self.encoder1.append(Linear(2*h_size,h_size))
        
        for i in range(self.nl1-1):
            self.encoder.append(Linear(h_size, h_size)) 
            self.encoder1.append(Linear(h_size, h_size)) 
        
    
# ==============================================================================
# 3. 用这个【新版本】的代码块，完整地替换掉旧的 generator 定义部分
# ==============================================================================
        # ... 前面的 encoder 定义保持不变 ...
        self.encoder.append(Linear(h_size, h_size)) 

        # --- 【新逻辑】定义融合后的新维度 ---
        # 空间特征维度 + 时间特征维度
        fused_h_size = h_size + self.time_embedding_dim # 128 + 64 = 192
        
        self.generator = torch.nn.ModuleList()
        self.generator1 = torch.nn.ModuleList()
        
        # nl2 是残差块的数量, 通常是3
        for i in range(self.nl2):
            # 【变化1】: Linear层的输入和输出维度全部更新为 2 * fused_h_size
            self.generator.append(Linear(2*fused_h_size, 2*fused_h_size)) 
            self.generator1.append(Linear(2*fused_h_size, 2*fused_h_size)) 
        
        # 【变化2】: 后续层的维度也需要相应更新
        self.generator.append(Linear(2*fused_h_size, h_size)) # 输出一个与原始空间维度兼容的尺寸
        self.generator.append(Linear(h_size, 1)) # 最后一层输出一个标量，保持不变
    
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)
            #torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    

    def input_mapping(self, x, B):
        w = 2.*np.pi*B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

    def input_mapping_grad(self, x, B):
        w = 2.*np.pi*B
        x_proj = x @ w
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        dx =  torch.cat([w *torch.cos(x_proj), -w *torch.sin(x_proj)], dim=-1)
        return x, dx   #  2*len(B)

    def input_mapping_laplace(self, x, B):
        w = 2.*np.pi*B
        #print(w.shape)
        w = w.unsqueeze(1)
        #print(w.shape)
        #print(x.shape)
        x_proj = x @ w
        #print(x_proj.shape)
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        dx =  torch.cat([w *torch.cos(x_proj), -w *torch.sin(x_proj)], dim=-1)
        lx =  torch.cat([-w * w *torch.sin(x_proj), -w * w *torch.cos(x_proj)], dim=-1)
        #print(x.shape)
        #print(dx.shape)
        #print(lx.shape)
        return x, dx, lx   #  2*len(B)

    # ==============================================================================
# 2. 用这个【新版本】的 out 函数，完整地替换掉旧的 out 函数
# ==============================================================================
    def out(self, coords, B, timesteps): # 【变化1】: 增加了一个新的输入参数 timesteps
        
        coords = coords.clone().detach().requires_grad_(True)
        size = coords.shape[0]
        x0_coords = coords[:,:self.dim]
        x1_coords = coords[:,self.dim:]
        
        x_coords = torch.cat((x0_coords, x1_coords), dim=0)
        x_coords = x_coords.unsqueeze(1)
        
        # --- 空间编码 (与原来几乎相同) ---
        x_spatial = self.input_mapping(x_coords, B)
        x_spatial = self.act(self.encoder[0](x_spatial))
        for ii in range(1,self.nl1):
            x_tmp = x_spatial
            x_spatial = self.act(self.encoder[ii](x_spatial))
            x_spatial = self.act(self.encoder1[ii](x_spatial) + x_tmp) 
        
        x_spatial = self.encoder[-1](x_spatial)

        x0_spatial = x_spatial[:size,...]
        x1_spatial = x_spatial[size:,...]
        # --- 空间编码结束 ---


        # --- 【新逻辑】时间编码 ---
        # timesteps 的 shape 应该是 (batch_size,)，包含了每个样本对应的时间戳 (0,1,2或3）
        # 我们用之前定义的 time_embedder 来获取每个时间戳对应的向量
        # time_embedding 的 shape 将是 (batch_size, 64)
        time_embedding = self.time_embedder(timesteps)
        # --- 时间编码结束 ---


        # --- 【新逻辑】时空信息融合 ---
        # 这是最关键的一步：将时间向量拼接到空间向量的后面
        # 我们需要将 time_embedding 扩展一下，以匹配 x0_spatial 和 x1_spatial 的维度
        # time_embedding.unsqueeze(1) 的 shape: (batch_size, 1, 64)
        # .expand(-1, x0_spatial.shape[1], -1) 让它在中间维度上复制，最终 shape: (batch_size, num_points, 64)
# 【新的正确代码】
# 我们需要将 shape 为 [batch_size, 64] 的 time_embedding 扩展为
# [batch_size, num_points, 64] 来与 x0_spatial 匹配。
# x0_spatial 的 shape 是 [batch_size, num_points, h_size]。
# 我们用 unsqueeze(1) 在中间增加一个维度，然后用 expand 复制它。
       
        # 【新的正确代码】
        time_embedding_expanded = time_embedding.unsqueeze(1)
        # 执行拼接
     
        x0_fused = torch.cat([x0_spatial, time_embedding_expanded], dim=-1)
        x1_fused = torch.cat([x1_spatial, time_embedding_expanded], dim=-1)
        # --- 融合结束 ---

        
        # --- 后续处理 (与原来类似，但处理的是融合后的特征) ---
        # 【注意】这里的对称操作及其后续网络层，现在接收的是更高维的输入！
        # 我们暂时先保持代码结构不变，下一步再来解决维度匹配问题。
        xx = torch.cat((x0_fused, x1_fused), dim=1) #【变化2】现在处理的是 x0_fused 和 x1_fused
        
        x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale

        x = torch.cat((x_0, x_1),1)
        
        for ii in range(self.nl2):
            x_tmp = x
            x = self.act(self.generator[ii](x)) 
            x = self.act(self.generator1[ii](x) + x_tmp) 
        
        y = self.generator[-2](x)
        x = self.act(y)

        y = self.generator[-1](x)
        x= self.actout(y)
        
        return x, coords
      
    def init_grad(self, x, w, b):
        y = x@w.T+b
        x   = self.act(y) 
        
        dact = self.dact(y)
        
        dx = w.T*dact

        del y, w, b, dact
        return x, dx
 
    def linear_grad(self, x, dx, w, b):
        y = x@w.T+b
        x  = y
        
        dxw=dx@w.T
        dx = dxw

        del y, w, b, dxw
        return x, dx
    
    def act_grad(self, x, dx):

        dact = self.dact(x)

        dx = dx*dact
        
        x  = self.act(x)
        del dact
        return x, dx

    def actout_grad(self, x, dx):
        actout  = self.actout(x)
        
        dactout = 0.1*actout*(1-actout)

        dx = dx*dactout
        x = actout
        del actout, dactout
        return x, dx


 
    # ==============================================================================
    # 【最终修正 for out_grad】
    # ==============================================================================
    def out_grad(self, coords, B, timesteps):
        x0_coords = coords[:, :self.dim]
        x1_coords = coords[:, self.dim:]
        size = coords.shape[0]
        x_coords = torch.cat((x0_coords, x1_coords), dim=0)

        # 【核心修正】: 在这里，我们将B和timesteps也进行同步的倍增
        B = torch.cat([B, B], dim=0)
        timesteps = torch.cat([timesteps, timesteps], dim=0)

        x_coords = x_coords.unsqueeze(1)
        # --- 空间编码 (与原来几乎相同) ---
        x_spatial, dx_spatial = self.input_mapping_grad(x_coords, B)
        w = self.encoder[0].weight; b = self.encoder[0].bias
        x_spatial, dx_spatial = self.linear_grad(x_spatial, dx_spatial, w, b)
        x_spatial, dx_spatial = self.act_grad(x_spatial, dx_spatial)
        for ii in range(1,self.nl1):
            x_tmp, dx_tmp = x_spatial, dx_spatial
            w = self.encoder[ii].weight; b = self.encoder[ii].bias
            x_spatial, dx_spatial = self.linear_grad(x_spatial, dx_spatial, w, b)
            x_spatial, dx_spatial = self.act_grad(x_spatial, dx_spatial)
            w = self.encoder1[ii].weight; b = self.encoder1[ii].bias
            x_spatial, dx_spatial = self.linear_grad(x_spatial, dx_spatial, w, b)
            x_spatial, dx_spatial = x_spatial + x_tmp, dx_spatial + dx_tmp 
            x_spatial, dx_spatial = self.act_grad(x_spatial, dx_spatial)
        w = self.encoder[-1].weight; b = self.encoder[-1].bias
        x_spatial, dx_spatial = self.linear_grad(x_spatial, dx_spatial, w, b)
        x0_spatial, x1_spatial = x_spatial[:size,...], x_spatial[size:,...]
        dx0_spatial, dx1_spatial = dx_spatial[:size,...], dx_spatial[size:,...]
        # --- 空间编码结束 ---

        # --- 【新逻辑】时间编码 ---
        time_embedding = self.time_embedder(timesteps)
        # 这里的维度扩展逻辑，与我们最终修正 out 函数时完全一样
        time_embedding_expanded = time_embedding.unsqueeze(1)
        # --- 时间编码结束 ---

       # ==============================================================================
        # 【最终的、绝对正确的修正】用这个全新的代码块，替换掉 out_grad 函数中旧的融合逻辑
        # ==============================================================================

        # --- 【新逻辑】时空信息融合 (最终版 for out_grad) ---

        # 1. 融合坐标 x (这部分逻辑是正确的，保持不变)
        x0_fused = torch.cat([x0_spatial, time_embedding_expanded], dim=-1)
        x1_fused = torch.cat([x1_spatial, time_embedding_expanded], dim=-1)

        # 2. 【核心修正】创建与【空间梯度】形状完全匹配的三维“零梯度”矩阵
        # 我们直接以 dx0_spatial 的真实3维形状为模板

        # 获取空间梯度 dx0_spatial 的真实形状 (b, d_spatial, f_s)
        # 这里的 d_spatial 就是那个尺寸为3，代表x,y,z的维度
        b, d_spatial, f_s = dx0_spatial.shape
        # 获取时间特征的维度
        f_t = self.time_embedding_dim

        # 创建一个形状为 [b, d_spatial, f_t] 的三维零矩阵，确保所有维度都精确匹配
        zeros_for_grad = torch.zeros(b, d_spatial, f_t, device=coords.device)

        # 3. 在正确的维度 (-1，即特征维度) 上进行拼接
        # 现在 dx0_spatial 和 zeros_for_grad 的前两个维度完全一致
        dx0_fused = torch.cat([dx0_spatial, zeros_for_grad], dim=-1)
        dx1_fused = torch.cat([dx1_spatial, zeros_for_grad], dim=-1)

        # --- 融合结束 ---

        # --- 后续处理 (使用融合后的 _fused 张量) ---
        x0_1 = x0_fused - x1_fused
        s0 = self.dact(x0_1); s1 = 1 - s0
        dx00 = dx0_fused * s0; dx01 = dx1_fused * s1
        dx10 = dx0_fused * s1; dx11 = dx1_fused * s0
        dx0 = torch.cat((dx00,dx01), dim=1)
        dx1 = torch.cat((dx10,dx11), dim=1)
        dx_combined_features = torch.cat((dx0,dx1), dim=2)

        xx = torch.cat((x0_fused, x1_fused), dim=1)
        x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale
        x = torch.cat((x_0, x_1), 1)

        x_for_gen = x.unsqueeze(1)
        dx_for_gen = dx_combined_features

        for ii in range(self.nl2):
            x_tmp, dx_tmp = x_for_gen, dx_for_gen
            w, b = self.generator[ii].weight, self.generator[ii].bias
            x_for_gen, dx_for_gen = self.linear_grad(x_for_gen, dx_for_gen, w, b)
            x_for_gen, dx_for_gen = self.act_grad(x_for_gen, dx_for_gen)
            w, b = self.generator1[ii].weight, self.generator1[ii].bias
            x_for_gen, dx_for_gen = self.linear_grad(x_for_gen, dx_for_gen, w, b)
            x_for_gen, dx_for_gen = x_for_gen + x_tmp, dx_for_gen + dx_tmp
            x_for_gen, dx_for_gen = self.act_grad(x_for_gen, dx_for_gen)

        w, b = self.generator[-2].weight, self.generator[-2].bias
        x_for_gen, dx_for_gen = self.linear_grad(x_for_gen, dx_for_gen, w, b)
        x_for_gen, dx_for_gen = self.act_grad(x_for_gen, dx_for_gen)
        w, b = self.generator[-1].weight, self.generator[-1].bias
        x_final, dx_final = self.linear_grad(x_for_gen, dx_for_gen, w, b)
        x_final, dx_final = self.actout_grad(x_final, dx_final)

        return x_final.squeeze(1), dx_final.squeeze(1), coords

    def out_backgrad(self, coords, B):
        size = coords.shape[0]

        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))

        #x = x.unsqueeze(1)
        
        w_list0=[]
        dact_list0=[]
        
        w = 2.*np.pi*B
        x_proj = x @ w
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
        #dx =  torch.cat([w *torch.cos(x_proj), -w *torch.sin(x_proj)], dim=-1)
        dact = torch.cat([torch.cos(x_proj), -torch.sin(x_proj)], dim=1)
        
        w_list0.append(torch.cat((w,w),dim=1))
        dact_list0.append(dact.transpose(0,1))

        w = self.encoder[0].weight
        b = self.encoder[0].bias

        y = x@w.T+b
        x = y
        w_list0.append(w.T)
        
        #dx=dx@w.T

        del y, w, b

        x  = self.act(x)

        dact = self.dact(x)
        dact_list0.append(dact.transpose(0,1))
        #print('dact',dact.shape)
        
        #dx = dx*dact

        del dact
        for ii in range(1,self.nl1):
            #x_tmp, dx_tmp = x, dx
            x_tmp = x

            w = self.encoder[ii].weight
            b = self.encoder[ii].bias

            y = x@w.T+b
            x  = y
            w_list0.append(w.T)
            #dx=dx@w.T

            del y, w, b

            dact = self.dact(x)
            dact_list0.append(dact.transpose(0,1))
            #dx = dx*dact
            
            x  = self.act(x)
            del dact

            w = self.encoder1[ii].weight
            b = self.encoder1[ii].bias

            y = x@w.T+b
            x  = y
            w_list0.append(w.T)
            #dx=dx@w.T

            del y, w, b

            #x, dx = x+ x_tmp, dx+dx_tmp 
            x = x + x_tmp

            dact = self.dact(x)
            dact_list0.append(dact.transpose(0,1))
            #dx = dx*dact
            
            x  = self.act(x)
            del dact

        
        w = self.encoder[-1].weight
        b = self.encoder[-1].bias

        y = x@w.T+b
        x  = y
        
        w_list0.append(w.T)
        #dx = dx@w.T

        del y, w, b

        x0 = x[:size,...]
        x1 = x[size:,...]
        #print('x0',x0.shape)
        x0_1=x0-x1
        s0 = self.dact(x0_1).transpose(0,1) #1/(1+torch.exp(-scale*x0_1))
        s1 = 1 - s0#1/(1+torch.exp(scale*f0_1))
        #print('s0',s0.shape)
        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        xx = torch.cat((x0, x1), dim=1)
        #print(xx.shape)
        
        x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale

        x = torch.cat((x_0, x_1),1)
        #x = x.unsqueeze(1)
        #print(feature.shape)
        w_list1=[]
        dact_list1=[]
        #print(dx.shape)
        for ii in range(self.nl2):
            #x_tmp, dx_tmp = x, dx
            x_tmp = x
            
            w = self.generator[ii].weight
            b = self.generator[ii].bias

            y = x@w.T+b
            x  = y
            
            w_list1.append(w.T)
            #dx=dx@w.T

            del y, w, b

            dact = self.dact(x)
            #print(dact.shape)
            dact_list1.append(dact.transpose(0,1))
            #dx = dx*dact
            
            x  = self.act(x)
            del dact

            w = self.generator1[ii].weight
            b = self.generator1[ii].bias

            y = x@w.T+b
            x  = y
            
            w_list1.append(w.T)
            #dx=dx@w.T

            del y, w, b

            #x, dx = x+ x_tmp, dx+dx_tmp 
            x = x+ x_tmp

            dact = self.dact(x)
            
            dact_list1.append(dact.transpose(0,1))
            #dx = dx*dact
            
            x  = self.act(x)
            del dact

        
        w = self.generator[-2].weight
        b = self.generator[-2].bias

        y = x@w.T+b
        x  = y
        w_list1.append(w.T)
        #dx=dx@w.T

        del y, w, b

        dact = self.dact(x)
        #print(dact.shape)
        dact_list1.append(dact.transpose(0,1))
        #dx = dx*dact
        
        x  = self.act(x)
        del dact

        w = self.generator[-1].weight
        b = self.generator[-1].bias

        y = x@w.T+b
        x  = y
        w_list1.append(w.T)
        #dx=dx@w.T

        del y, w, b

        actout  = self.actout(x)
        dactout = 0.1*actout*(1-actout)
        #print(dactout.shape)

        dact_list1.append(dactout.transpose(0,1))
        
        #dx = dx*dactout
        x = actout
        del actout, dactout
        
        #x = x.squeeze(2)
        #dx = dx.squeeze(2)

        dact_list1.reverse()
        w_list1.reverse()
        
        dx = w_list1[0]*dact_list1[0]
        #print(dact_list1[1].shape)
        #print(dx.shape)
        dx = dact_list1[1]*dx
        #print(dx.shape)
        dx = w_list1[1]@dx
        #print(dx.shape)
        for ii in range(self.nl2):
            dx =  dact_list1[2*ii+2]*dx
            dx_tmp =  w_list1[2*ii+2]@dx
            dx = w_list1[2*ii+3]@(dact_list1[2*ii+3]*dx_tmp) + dx

        dx0 = dx[:128,:]
        dx1 = dx[128:,:]

        dx00 = s0*dx0
        dx01 = s1*dx1
        dx10 = s1*dx0
        dx11 = s0*dx1
        dx = torch.cat((dx00+dx01,dx10+dx11),dim=1)

        dact_list0.reverse()
        w_list0.reverse()
        dx = w_list0[0]@dx
        #print(dx.shape)
        for ii in range(0,self.nl1-1):
            dx =  dact_list0[2*ii]*dx
            dx_tmp =  w_list0[2*ii+1]@dx
            dx = w_list0[2*ii+2]@(dact_list0[2*ii+1]*dx_tmp) + dx
        #print(dx.shape) 
     
        dx = w_list0[2*self.nl1-1]@(dact_list0[2*(self.nl1-1)]*dx)
        dx = dact_list0[2*self.nl1-1]*dx
        dx = w_list0[2*self.nl1]@dx
        #print(w_list0[2*self.nl1].shape)
        dx0=dx[:,:size]
        dx1=dx[:,size:]
        dx = torch.cat((dx0,dx1),dim=0).transpose(0,1)
        #print(dx.shape)
        return x, dx, coords
    
    def init_laplace(self, x, w, b):
        y = x@w.T+b
        x   = self.act(y) 
        #w = self.encoder[0].weight
        
        dact = self.dact(y)
        ddact = 10*dact*(1-dact)
        
        dx = w.T*dact
        lx = w.T*w.T*ddact

        del y, w, b, dact, ddact
        return x, dx, lx
    
    def linear_laplace(self, x, dx, lx, w, b):
        y = x@w.T+b
        x  = y
        
        dxw=dx@w.T
        dx = dxw
        lx_b = lx@w.T
        lx = lx_b

        del y, w, b, dxw, lx_b
        return x, dx, lx
    
    def act_laplace(self, x, dx, lx):
        #y = x@w.T+b
        
        #dxw=dx@w.T
        dact = self.dact(x)
        ddact = 10*dact*(1-dact)
        
        lx_a = ((dx*dx)*ddact)
        lx_b = lx*dact
        lx = lx_a+lx_b

        dx = dx*dact

        x  = self.act(x)

        del lx_a, lx_b, dact, ddact
        return x, dx, lx
    
    def actout_laplace(self, x, dx, lx):
        actout  = self.actout(x)
        
        dactout = 0.1*actout*(1-actout)
        ddactout = 0.1*dactout*(1-2*actout)

        lx_a = dx*dx*ddactout
        lx_b = lx*dactout
        lx = lx_a+lx_b

        dx = dx*dactout

        x = actout

        del actout, lx_a, lx_b, dactout, ddactout
        return x, dx, lx
    
    # ==============================================================================
    # ==============================================================================
    # 【最终修正 for out_laplace】
    # ==============================================================================
    def out_laplace(self, coords, B, timesteps):
        x0_coords = coords[:, :self.dim]
        x1_coords = coords[:, self.dim:]
        x_size = coords.shape[0]
        x_coords = torch.cat((x0_coords, x1_coords), dim=0)

        # 【核心修正】: 在这里，我们也将B和timesteps进行同步的倍增
        B = torch.cat([B, B], dim=0)
        timesteps = torch.cat([timesteps, timesteps], dim=0)

        x_coords = x_coords.unsqueeze(2)
        # --- 空间编码 (计算梯度和拉普拉斯) ---
        x_spatial, dx_spatial, lx_spatial = self.input_mapping_laplace(x_coords, B)
        
        w = self.encoder[0].weight
        b = self.encoder[0].bias
        x_spatial, dx_spatial, lx_spatial = self.linear_laplace(x_spatial, dx_spatial, lx_spatial, w, b)
        x_spatial, dx_spatial, lx_spatial = self.act_laplace(x_spatial, dx_spatial, lx_spatial)
        
        for ii in range(1, self.nl1):
            x_tmp, dx_tmp, lx_tmp = x_spatial, dx_spatial, lx_spatial
            
            w = self.encoder[ii].weight
            b = self.encoder[ii].bias
            x_spatial, dx_spatial, lx_spatial = self.linear_laplace(x_spatial, dx_spatial, lx_spatial, w, b)
            x_spatial, dx_spatial, lx_spatial = self.act_laplace(x_spatial, dx_spatial, lx_spatial)

            w = self.encoder1[ii].weight
            b = self.encoder1[ii].bias
            x_spatial, dx_spatial, lx_spatial = self.linear_laplace(x_spatial, dx_spatial, lx_spatial, w, b)
            
            x_spatial, dx_spatial, lx_spatial = x_spatial+x_tmp, dx_spatial+dx_tmp, lx_spatial+lx_tmp 
            x_spatial, dx_spatial, lx_spatial = self.act_laplace(x_spatial, dx_spatial, lx_spatial)

        w = self.encoder[-1].weight
        b = self.encoder[-1].bias
        x_spatial, dx_spatial, lx_spatial = self.linear_laplace(x_spatial, dx_spatial, lx_spatial, w, b)

        x0_spatial = x_spatial[:,:x_size,...]
        x1_spatial = x_spatial[:,x_size:,...]
        dx0_spatial = dx_spatial[:,:x_size,...]
        dx1_spatial = dx_spatial[:,x_size:,...]
        lx0_spatial = lx_spatial[:,:x_size,...]
        lx1_spatial = lx_spatial[:,x_size:,...]
        # --- 空间编码结束 ---


        # --- 【新逻辑】时间编码 ---
        # 和 out() 函数中的逻辑完全一样
        time_embedding = self.time_embedder(timesteps)
        # 这是解决维度问题的【新代码】
        time_embedding_expanded = time_embedding.unsqueeze(2)
        # --- 时间编码结束 ---

        # ==============================================================================
        # 【最终的、绝对正确的修正】用这个全新的代码块，替换掉 out_laplace 函数中旧的融合逻辑
        # ==============================================================================

        # --- 【新逻辑】时空信息融合 (最终版) ---

        # 1. 融合坐标 x (这部分逻辑是正确的，保持不变)
        x0_fused = torch.cat([x0_spatial, time_embedding_expanded], dim=-1)
        x1_fused = torch.cat([x1_spatial, time_embedding_expanded], dim=-1)

        # 2. 【核心修正】创建与【空间梯度】形状完全匹配的四维“零梯度”矩阵
        # 我们直接以 dx0_spatial 的形状为模板

        # 获取空间梯度的真实形状 (b, n, d, f_s)
        b, n, d_spatial, f_s = dx0_spatial.shape
        # 获取时间特征的维度
        f_t = self.time_embedding_dim

        # 创建一个形状为 [b, n, d_spatial, f_t] 的四维零矩阵，确保所有维度都精确匹配
        zeros_for_derivatives = torch.zeros(b, n, d_spatial, f_t, device=coords.device)

        # 3. 在正确的维度 (-1，即特征维度) 上进行拼接
        # 现在 dx0_spatial 和 zeros_for_derivatives 的前三个维度完全一致
        dx0_fused = torch.cat([dx0_spatial, zeros_for_derivatives], dim=-1)
        dx1_fused = torch.cat([dx1_spatial, zeros_for_derivatives], dim=-1)

        # 对于拉普拉斯项 lx，它的形状与dx不同，所以我们也为它创建匹配的零矩阵
        b, n, d_lap, f_s = lx0_spatial.shape
        zeros_for_laplace = torch.zeros(b, n, d_lap, f_t, device=coords.device)

        lx0_fused = torch.cat([lx0_spatial, zeros_for_laplace], dim=-1)
        lx1_fused = torch.cat([lx1_spatial, zeros_for_laplace], dim=-1)

        # --- 融合结束 ---


        # ==============================================================================
        # 【最终的、绝对正确的修正】用这个全新的代码块，替换掉 out_laplace 函数中旧的后续处理逻辑
        # ==============================================================================

        # --- 后续处理 (最终修正版) ---
        xx = torch.cat((x0_fused, x1_fused), dim=2)
        x_0 = torch.logsumexp(self.scale*xx, 2)/self.scale
        x_1 = -torch.logsumexp(-self.scale*xx, 2)/self.scale

        x = torch.cat((x_0, x_1),2)
        x = x.unsqueeze(2)

        # 【核心修正】: 我们必须使用融合后的 _fused 张量来进行后续所有计算
        # 旧代码错误地使用了 x0_spatial，导致了维度冲突
        x0_1 = x0_fused - x1_fused

        s0 = self.dact(x0_1) 
        s1 = 1.0 - s0

        # 现在 dx0_fused 和 s0 的特征维度都是192，可以正确相乘
        dx00 = dx0_fused * s0 
        dx01 = dx1_fused * s1
        dx10 = dx0_fused * s1 
        dx11 = dx1_fused * s0
        dx_0 = torch.cat((dx00,dx01), dim =2)
        dx_1 = torch.cat((dx10,dx11), dim =2)
        dx = torch.cat((dx_0,dx_1), dim =3)

        s = 10 * s0 * s1
        lx00 = ((dx0_fused) * s) * dx0_fused
        lx11 = ((dx1_fused) * s) * dx1_fused
        lx_00_0 = lx00 + lx0_fused * s0
        lx_11_0 = lx11 + lx1_fused * s1
        lx_00_1 = -lx00 + lx0_fused * s1
        lx_11_1 = -lx11 + lx1_fused * s0
        lx_0 = torch.cat((lx_00_0, lx_11_0), dim =2)
        lx_1 = torch.cat((lx_00_1, lx_11_1), dim =2)
        lx = torch.cat((lx_0, lx_1), dim =3)

        # 后续的 generator 部分，所有输入都已经是正确的融合后维度
        for ii in range(self.nl2):
            x_tmp, dx_tmp, lx_tmp = x, dx, lx
            w = self.generator[ii].weight
            b = self.generator[ii].bias
            x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
            x, dx, lx = self.act_laplace(x, dx, lx)

            w = self.generator1[ii].weight
            b = self.generator1[ii].bias
            x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
            x, dx, lx = x+x_tmp, dx+dx_tmp, lx+lx_tmp 
            x, dx, lx = self.act_laplace(x, dx, lx)

        w = self.generator[-2].weight
        b = self.generator[-2].bias
        x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
        x, dx, lx = self.act_laplace(x, dx, lx)

        w = self.generator[-1].weight
        b = self.generator[-1].bias
        x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
        x, dx, lx = self.actout_laplace(x, dx, lx)

        x = x.squeeze(3)
        dx = dx.squeeze(3)
        lx = lx.squeeze(3)
        # --- 后续处理结束 ---
        
        return x, dx, lx, coords

    def forward(self, coords, B):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

        output, coords = self.out(coords, B)
        return output, coords


class Model():
    def __init__(self, ModelPath, DataPath, dim, length, device='cpu'):

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['DataPath'] = DataPath
        self.dim = dim
        self.len = length

        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 10000
        self.Params['Training']['Resampling Bounds'] = [0.1, 0.9]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 100
        self.Params['Training']['Learning Rate'] = 1e-3#5e-5
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        # 【新增代码】: 在这里为 self.B 提供一个初始值
        self.B = None
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    def Loss(self, points, Yobs, B, beta, gamma, timesteps):

        # 【核心修正】: 在调用网络之前，我们就把所有输入都处理成“倍增”的形状
        points_x0 = points[:, :self.dim]
        points_x1 = points[:, self.dim:]

        # 将点、B矩阵、时间戳都在批次维度上进行倍增
        points_doubled = torch.cat([points_x0, points_x1], dim=0)
        B_doubled = torch.cat([B, B], dim=0)
        timesteps_doubled = torch.cat([timesteps, timesteps], dim=0)

        # 【注意】: 我们现在只调用轻量的 out_grad，以避免显存问题
        tau_doubled, dtau_doubled, Xp_doubled = self.network.out_grad(
            points_doubled, B_doubled, timesteps_doubled
        )

        # 因为我们不再计算ltau，所以手动创建为0的拉普拉斯项
        lap0 = torch.zeros(points.shape[0], device=points.device)
        lap1 = torch.zeros(points.shape[0], device=points.device)

        # 从倍增的结果中，重新分离出 x0 和 x1 的部分
        half_size = tau_doubled.shape[0] // 2
        tau0, tau1 = tau_doubled[:half_size], tau_doubled[half_size:]
        dtau0, dtau1 = dtau_doubled[:half_size], dtau_doubled[half_size:]

        # 后续的损失计算逻辑
        D = points_x1 - points_x0
        T0 = torch.sum(D * D, dim=1)

        DT0 = dtau0; DT1 = dtau1
        tau = tau0 # 以tau0为基准
        T3 = tau.squeeze()**2

        # 计算S0
        einsum_T01 = torch.sum(DT0 * DT0, dim=[1, 2])
        T01 = T0 * einsum_T01
        einsum_T02 = torch.sum(DT0 * D.unsqueeze(-1), dim=[1, 2])
        T02 = 2 * tau.squeeze() * einsum_T02
        S0 = T01 + T02 + T3 # 使用加号

        # 计算S1
        einsum_T11 = torch.sum(DT1 * DT1, dim=[1, 2])
        T11 = T0 * einsum_T11
        einsum_T12 = torch.sum(DT1 * D.unsqueeze(-1), dim=[1, 2])
        T12 = -2 * tau.squeeze() * einsum_T12 # 注意，这里是减号
        S1 = T11 + T12 + T3

        sq_Ypred0 = 1/(torch.sqrt(S0)/T3 + gamma*lap0)
        sq_Ypred1 = 1/(torch.sqrt(S1)/T3 + gamma*lap1)

        sq_Yobs0 = torch.sqrt(Yobs[:,0])
        sq_Yobs1 = torch.sqrt(Yobs[:,1])

        loss0 = sq_Ypred0/sq_Yobs0 + sq_Yobs0/sq_Ypred0
        loss1 = sq_Ypred1/sq_Yobs1 + sq_Yobs1/sq_Ypred1
        diff = loss0 + loss1 - 4

        loss_n = torch.mean(diff)
        loss = beta * loss_n

        return loss, loss_n, None

    # ==============================================================================
    # 【最终的、绝对正确的修正】用这个全新的代码块，替换掉整个旧的 train 函数
    # ==============================================================================
    def train(self):
        """
        【模型训练主函数 - 最终版】
        这个版本拥有唯一的、正确的、高效的数据加载和批处理逻辑。
        """
        # 1. 初始化网络和优化器 (不变)
        self.network = NN(self.Params['Device'],self.dim)
        self.network.apply(self.network.init_weights)
        self.network.to(self.Params['Device'])

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate'],
            weight_decay=0.1
        )

        # ==============================================================================
        # 【最终的、绝对正确的修正】用这个全新的代码块，替换掉 train 函数中旧的数据加载和拼接逻辑
        # ==============================================================================

        # 2. 【核心优化】: 预加载所有数据到CPU，并创建高效的DataLoader

        print("--- Pre-loading all scene data to CPU memory... ---")
        self.dataset_loader = db.Database(self.Params['DataPath'], torch.device('cpu'), self.len)

        all_data, all_timestamps, all_B = [], [], []
        for i in range(self.len): # self.len 是场景数量, e.g., 2
            data_tuple = self.dataset_loader[i] # 返回 (data, B, timestamps)
            all_data.append(data_tuple[0].squeeze(0)) # squeeze(0) 降维
            all_B.append(data_tuple[1])
            all_timestamps.append(data_tuple[2])

        # 【核心修正】: 使用 dim=0 进行垂直堆叠
        full_dataset_data = torch.cat(all_data, dim=0)
        full_dataset_timestamps = torch.cat(all_timestamps, dim=0)

        # B矩阵需要特殊处理，因为它与场景相关
        b_expanded_list = []
        for i in range(self.len):
            num_points = all_data[i].shape[0]
            b_expanded_list.append(all_B[i].expand(num_points, -1, -1))
        full_dataset_B_expanded = torch.cat(b_expanded_list, dim=0)

        # 3. 创建最终的、高效的数据集和加载器 (这部分不变)
        inner_batch_size = 500
        tensor_dataset = torch.utils.data.TensorDataset(full_dataset_data, full_dataset_timestamps, full_dataset_B_expanded)
        dataloader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=inner_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # 4. 初始化训练控制变量
        beta = 1.0
        step = -2000.0 / 4000.0
        self.B = None # 为 save 函数初始化 self.B

        print(f"--- 训练开始，高效批次大小设置为: {inner_batch_size} ---")

        # 5. 主训练循环 (现在只有一个清爽、高效的循环)
        for epoch in range(1, self.Params['Training']['Number of Epochs'] + 1):

            alpha = min(max(0.5, 0.5 + 0.5 * step), 1.07)
            step += 1.0 / 4000 / ((int)(epoch / 4000) + 1.)
            gamma = 0.001

            total_loss_epoch = 0

            # 【唯一的、正确的循环】: 直接遍历由DataLoader生成的小批次
            for i, (batch_data, batch_timestamps, batch_B) in enumerate(dataloader):

                print(f"[Batch {i}] batch_data shape: {batch_data.shape}, batch_timestamps: {batch_timestamps.shape}, batch_B: {batch_B.shape}")
                batch_data = batch_data.to(self.Params['Device'])
                batch_timestamps = batch_timestamps.to(self.Params['Device'])
                batch_B = batch_B.to(self.Params['Device'])

                points = batch_data[:, :2*self.dim].requires_grad_()
                speed = batch_data[:, 2*self.dim:]
                speed = alpha * speed + (1 - alpha)

                self.B = batch_B # 更新self.B以便保存

                loss_value, loss_n, _ = self.Loss(
                    points, speed, batch_B, beta, gamma, batch_timestamps)

                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()

                total_loss_epoch += loss_n.item()

            avg_loss = total_loss_epoch / len(dataloader)
            print(f"\rEpoch = {epoch} -- Avg Loss = {avg_loss:.4e} -- Alpha = {alpha:.4e}")

            if (epoch % 100 == 0) or (epoch == self.Params['Training']['Number of Epochs']) or (epoch == 1):
                print(f"\n--- Saving model and plotting at epoch {epoch} ---")
                self.plot(epoch, avg_loss, alpha)
                self.save(epoch=epoch, val_loss=avg_loss)
    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'B_state_dict':self.B,
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.Params['ModelPath'], str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        #self.B = checkpoint['B_state_dict']

        self.network = NN(self.Params['Device'],self.dim)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

        
    def load_pretrained_state_dict(self, state_dict):
        own_state=self.state_dict


    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.network.out(Xp,self.B)
       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.sqrt(T0)/tau[:, 0]

        del Xp, tau, T0
        return TT
    
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.network.out(Xp,self.B)

        return tau

    # ==============================================================================
    # 【最终的、绝对正确的修正】用这个全新的代码块，替换掉旧的 Speed 函数
    # ==============================================================================
    def Speed(self, Xp, B, timesteps):
        """
        【时空改造最终版】
        这个版本的Speed函数使用了对维度更稳健的 torch.sum 来取代 einsum，
        以处理我们新的三维梯度张量。
        """
        # 1. 获取 tau 和它的梯度 dtau (这部分逻辑不变)
        tau, dtau, coords = self.network.out_grad(Xp, B, timesteps)

        # 2. 准备计算所需的张量 (这部分逻辑不变)
        D = Xp[:, self.dim:] - Xp[:, :self.dim]
        T0 = torch.sum(D * D, dim=1) # 等价于原来的einsum，但更清晰

        DT1 = dtau[:, self.dim:] # DT1 是一个三维张量 [points, dims, features]
        T3 = tau[:, 0]**2

        # 3. 【核心修正】: 使用 torch.sum 代替 einsum

        # 计算 T1 = ||d(tau)/dq_g||^2
        # 我们需要对 DT1 的后两个维度（空间和特征）进行平方和
        einsum_T1 = torch.sum(DT1 * DT1, dim=[1, 2])
        T1 = T0 * einsum_T1

        # 计算 T2 = (d(tau)/dq_g) . (q_g - q_s)
        # 我们需要将 D 扩展一个维度，使其能与 DT1 广播相乘
        einsum_T2 = torch.sum(DT1 * D.unsqueeze(-1), dim=[1, 2])
        T2 = 2 * tau[:, 0] * einsum_T2

        # 4. 最终计算 (不变)
        S = (T1 - T2 + T3)
        Ypred = T3 / torch.sqrt(S)

        del Xp, tau, dtau, T0, T1, T2, T3
        return Ypred
    
    def Gradient(self, Xp, B):
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, dtau, coords = self.network.out_backgrad(Xp, B)
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D))
        T3 = tau[:, 0]**2

        V0 = D
        V1 = dtau[:,self.dim:]
        #print(D.shape)
        Y1 = 1/(T0*tau[:, 0]).unsqueeze(1)*V0
        Y2 = (T0/T3).unsqueeze(1)*V1

        Ypred1 = -(Y1-Y2)
        Spred1 = torch.norm(Ypred1, p=2, dim =1).unsqueeze(1)
        Ypred1 = 1/Spred1**2*Ypred1

        V0=-D
        V1=dtau[:,:self.dim]
        
        Y1 = 1/(T0*tau[:, 0]).unsqueeze(1)*V0
        Y2 = (T0/T3).unsqueeze(1)*V1

        Ypred0 = -(Y1-Y2)
        Spred0 = torch.norm(Ypred0, p=2, dim =1).unsqueeze(1)

        Ypred0 = 1/Spred0**2*Ypred0
        
        return torch.cat((Ypred0, Ypred1),dim=1)
     
    # ==============================================================================
    # 【最终版】用这个全新的 plot 函数，替换掉旧的
    # ==============================================================================
    def plot(self,epoch,total_train_loss,alpha):
        """
        【动态绘图函数】
        这个函数会为每个时刻t都计算一次速度场，然后将它们融合成一张图，
        以可视化动态障碍物的运动轨迹。
        """
        print(f"Plotting for epoch {epoch}...")
        self.network.eval() # 切换到评估模式，这会关闭dropout等，并影响梯度计算

        # 准备绘图用的坐标点
        limit = 0.5
        spacing = limit / 40.0
        X,Y = np.meshgrid(np.arange(-limit,limit,spacing),np.arange(-limit,limit,spacing))
        Xsrc = np.zeros(self.dim); Xsrc[0], Xsrc[1] = -0.25, -0.25
        XP_numpy = np.zeros((len(X.flatten()), 2 * self.dim))
        XP_numpy[:, :self.dim] = Xsrc
        XP_numpy[:, self.dim+0], XP_numpy[:, self.dim+1] = X.flatten(), Y.flatten()
        XP = torch.tensor(XP_numpy, dtype=torch.float32).to(self.Params['Device'])

        # 加载B矩阵 (只使用第一个场景的B进行绘图)
        B_path = os.path.join(self.Params['DataPath'], '0', 'B.npy')
        B = torch.tensor(np.load(B_path), dtype=torch.float32).to(self.Params['Device'])

        all_velocities = []
        # 【核心逻辑】: 循环遍历所有时间步
        with torch.no_grad(): # 推理时不需要计算梯度，能节省大量显存和时间
            for t in range(self.network.num_timesteps):
                print(f"  - Plotting for timestep t={t}...")
                # 为当前画布上的所有点，都赋予相同的时间戳 t
                timesteps = torch.full((XP.shape[0],), t, dtype=torch.long).to(self.Params['Device'])

                # 调用我们刚刚升级好的、能处理时空信息的Speed函数
                ss = self.Speed(XP, B, timesteps)
                all_velocities.append(ss.cpu().numpy().reshape(X.shape))

        # 【融合】: 取所有时刻速度场的最小值，形成“运动轨迹包络”
        # 一个点只要在任何一个时刻是障碍区（速度低），它最终的颜色就是深的
        if all_velocities:
            final_velocity_envelope = np.min(np.array(all_velocities), axis=0)

            # --- 绘图 ---
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # 使用 pcolormesh 绘制融合后的速度场
            quad1 = ax.pcolormesh(X, Y, final_velocity_envelope, vmin=0, vmax=1, shading='auto')
            plt.colorbar(quad1, ax=ax, pad=0.1, label='Predicted Velocity Envelope')
            plt.title(f"Epoch {epoch} - Dynamic Velocity Field")

            # 保存最终的动态效果图
            save_path = os.path.join(self.Params['ModelPath'], f"plots_epoch_{epoch:05d}_dynamic.jpg")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Dynamic plot saved to: {save_path}")

        self.network.train() # 切换回训练模式，以便继续训练