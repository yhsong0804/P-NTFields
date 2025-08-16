#!/bin/bash
# 快速恢复脚本 - 重新连接后运行此脚本快速了解项目状态

echo "=== 动态障碍物PINN项目状态检查 ==="
echo ""

# 检查项目文件
echo "📁 项目文件检查:"
files_to_check=(
    "models/model_dynamic.py"
    "models/data_dynamic.py" 
    "dataprocessing/speed_sampling_dynamic.py"
    "train/train_dynamic.py"
    "test_dynamic_setup.py"
    "test_dynamic_system.py"
    "PROJECT_PROGRESS.md"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
    fi
done

echo ""
echo "📊 数据集状态:"

# 检查静态数据集
if [ -d "datasets/gibson_3smalltree/my_final_scene" ]; then
    echo "  ✅ 静态数据集存在"
    echo "    - 场景数量: $(ls datasets/gibson_3smalltree/my_final_scene/ | wc -l)"
else
    echo "  ❌ 静态数据集缺失"
fi

# 检查动态数据集目录
if [ -d "datasets/gibson_3smalltree_dynamic" ]; then
    echo "  ✅ 动态数据集目录已创建"
    
    # 检查时序OBJ文件
    obj_count=$(find datasets/gibson_3smalltree_dynamic -name "mesh_z_up_t*.obj" 2>/dev/null | wc -l)
    if [ $obj_count -gt 0 ]; then
        echo "    - 时序OBJ文件: $obj_count 个"
    else
        echo "    - ⚠️  尚未放入时序OBJ文件"
    fi
    
    # 检查预处理数据
    npy_count=$(find datasets/gibson_3smalltree_dynamic -name "*_t*.npy" 2>/dev/null | wc -l)
    if [ $npy_count -gt 0 ]; then
        echo "    - 预处理数据: $npy_count 个文件"
    else
        echo "    - ⚠️  尚未生成预处理数据"
    fi
else
    echo "  ❌ 动态数据集目录不存在"
fi

echo ""
echo "🎯 下一步操作:"
echo "1. 准备4个时刻的OBJ文件 (mesh_z_up_t0.obj ~ mesh_z_up_t3.obj)"
echo "2. 放入目录: datasets/gibson_3smalltree_dynamic/my_final_scene/0/"
echo "3. 运行: python test_dynamic_setup.py"
echo "4. 运行: python train/train_dynamic.py"

echo ""
echo "📋 快速命令:"
echo "  查看项目文档: cat PROJECT_PROGRESS.md"
echo "  测试系统状态: python test_dynamic_system.py"
echo "  开始训练: python train/train_dynamic.py"

echo ""
echo "=== 检查完成 ==="