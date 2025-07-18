from scipy.spatial import cKDTree as KDTree
import numpy as np
import os
import traceback
import igl
import open3d as o3d
#全局变量：KD树，网格点，配置（包含了所有参数）
kdtree, grid_points, cfg = None, None, None
#主函数
def voxelized_pointcloud_sampling(path):
    try:
        #获取输出目录和文件名
        out_path = os.path.dirname(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path,file_name + '_scaled.off')
        #输出npz文件名（体素分辨率和采样点数量信息）
        out_file = out_path + '/voxelized_point_cloud_{}res_{}points.npz'.format(cfg.input_res, cfg.num_points)

        #存在就跳过
        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            return



        #mesh = trimesh.load(input_file)
        #point_cloud = mesh.sample(cfg.num_points)
        '''
        v, f = igl.read_triangle_mesh(input_file)
        B, FI = igl.random_points_on_mesh(cfg.num_points,v,f)
        
        point_cloud=np.dot(np.diag(B[:,0]),v[f[FI,0],:]) + \
                    np.dot(np.diag(B[:,1]),v[f[FI,1],:]) + \
                    np.dot(np.diag(B[:,2]),v[f[FI,2],:])
        '''
        #点云采样：从mesh表面采样num_points个点
        mesh = o3d.io.read_triangle_mesh(input_file) #读mesh（o3d可直接读off格式）
        pcl = mesh.sample_points_poisson_disk(cfg.num_points)#泊松盘采样，较均匀
        #保存obj格式便于调试可视化
        point_cloud = np.asarray(pcl.points)#得到N*3点点云数组
        pcf=np.array([[0,0,0]])#dummmy法线（不重要）
        igl.write_obj(out_path + '/pc.obj', point_cloud, pcf)
        ''''''
        #print(point_cloud)
        #体素化标记：把所有采样点映射到grid上
        occupancies = np.zeros(len(grid_points), dtype=np.int8)#初始化所有网格点都“空”
        #用KD树实现
        _, idx = kdtree.query(point_cloud)#idx是每个点最近都grid点的索引
        occupancies[idx] = 1              #采中记为1（被占用）
        #压缩为二进制bits存储
        compressed_occupancies = np.packbits(occupancies)
        #保存所有关键信息到npz
        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = cfg.bb_min, bb_max = cfg.bb_max, res = cfg.input_res)
        print('Finished: {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))
#初始化全局KD树和网络
def init(cfg_param):
    global kdtree, grid_points, cfg
    cfg = cfg_param
    #根据AABB和体素分辨率生成所有网格点
    grid_points = create_grid_points_from_bounds(cfg.bb_min, cfg.bb_max, cfg.input_res)
    kdtree = KDTree(grid_points)#建立快速最近邻索引
#网格点生成函数
def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)#三个轴方向均匀采样res个点
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')#生成三维体素网格
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))#所有体素中心点三维坐标（N，3）
    del X, Y, Z, x
    return points_list