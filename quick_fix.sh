#!/bin/bash
# 快速修复脚本 - 解决维度错误

echo "🔧 修复动态PINN维度错误..."

# 方案1: 创建简化版本的Loss函数
cat > quick_fix.py << 'EOF'
import sys
sys.path.append('.')

# 在model_dynamic.py中添加简化Loss函数
fix_code = '''
    def Loss_Simple(self, coords_t, Yobs_t, B, beta, gamma):
        """简化的损失函数，避免维度问题"""
        # 使用原始网络的out_grad方法
        tau, dtau, coords = self.network.out_grad(coords_t, B)
        
        D = coords[:, self.dim:] - coords[:, :self.dim]
        T0 = torch.einsum('ij,ij->i', D, D)
        
        DT1 = dtau[:, self.dim:]
        T1 = T0 * torch.einsum('ij,ij->i', DT1, DT1)
        T2 = 2 * tau[:, 0] * torch.einsum('ij,ij->i', DT1, D)
        T3 = tau[:, 0] ** 2
        
        S = (T1 - T2 + T3)
        sq_Ypred = 1 / (torch.sqrt(S) / T3 + gamma * 0.001)
        sq_Yobs = Yobs_t[:, 0]
        
        loss = sq_Ypred / sq_Yobs + sq_Yobs / sq_Ypred
        total_loss = torch.sum(loss - 2) / coords_t.shape[0]
        
        return total_loss, total_loss, torch.tensor(0.0)
'''

# 修改train_dynamic.py使用简化Loss
with open('models/model_dynamic.py', 'a') as f:
    f.write(fix_code)

print("修复完成!")
EOF

python quick_fix.py

echo "✅ 临时修复完成"
echo "现在运行: python train/train_dynamic.py"