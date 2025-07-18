#!/usr/bin/env python3
# save_compare.py

#预处理——网格缩放——对比图
import os
import open3d as o3d

def main():
    # 1. 加载两幅网格，并做些简单处理
    mesh0 = o3d.io.read_triangle_mesh('datasets/gibson/0/mesh_z_up.obj')
    mesh0.compute_vertex_normals()
    mesh0.paint_uniform_color([1.0, 0.0, 0.0])   # 红色

    mesh1 = o3d.io.read_triangle_mesh('datasets/gibson/0/mesh_z_up_scaled.off')
    mesh1.compute_vertex_normals()
    mesh1.paint_uniform_color([0.0, 1.0, 0.0])   # 绿色

    # 把第一个网格向左平移一点，便于并排对比
    mesh0.translate((-1.2, 0.0, 0.0))

    # 2. 创建一个可视化器（不显示交互窗口）
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, visible=False)

    # 3. 把两幅网格添加到场景中
    vis.add_geometry(mesh0)
    vis.add_geometry(mesh1)

    # 4. 渲染更新
    vis.poll_events()
    vis.update_renderer()

    # 5. 确保输出目录存在
    out_dir = '/workspace/P-NTFields/gpt'
    os.makedirs(out_dir, exist_ok=True)

    # 6. 截图并保存
    output_path = os.path.join(out_dir, 'comparison.png')
    success = vis.capture_screen_image(output_path, do_render=True)
    if success:
        print(f"Saved screenshot to {output_path}")
    else:
        print("Failed to capture screenshot")

    # 7. 关闭窗口
    vis.destroy_window()

if __name__ == "__main__":
    main()
