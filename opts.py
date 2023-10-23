def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # 本次实验的名称,作为log中文件夹的名字
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    # 输出目录
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    # 指定数据集的目录
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    # 粗网络全连接的层数，一般可以更浅更窄
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    # 网络宽度
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')

    # 精细网络的全连接层数
    # 默认精细网络的深度和宽度与粗糙网络是相同的
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')

    # 这里的batch size，指的是光线的数量,像素点的数量
    # N_rand 配置文件中是1024
    # 32*32*4=4096
    # 800*800/4096=156 这里由于一般Nerf会对合成数据集进行下采样编程400*400，所以config下的具体配置都是取的光线数量为1024即可保证比例156不变(即每次训练1024条光线/像素点，总共训练一次照片需要156batch，当然也可以取别的值)
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    # 学习率
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 学习率衰减
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')

    # 如果batch_size还是很大，那么这里会对batch进一步分批处理，当然观察config一般设置的batch_size会小于chunk因此用不到，这里主要是为了保护batch_size过大造成内存泄漏（注意这里是对光线进行分批次），具体取值取决于内存大小
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # 网络中处理的点的数量，以防一条光线上的点数量过多造成内存泄漏（因此光线数量受chunk限制，一条光线的采样点受netchunk限制都是为了避免内存泄漏），具体取值取决于内存大小
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    # 合成的数据集一般都是True, 每次只从一堆张图片中(可以理解为整个数据集)选取随机光线1024条就可以，合成数据集不需要所有像素点引射线参与训练，因此合成数据集的视角包含许多重复的信息(360°拍照)造成梯度更新太小
    # 真实的数据集一般都是False, 即真实场景一般必须全部图像都采样1024个像素点引射线参与训练才能准确还原场景
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')

    # 不加载权重
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # 粗网络的权重文件的位置
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options，一条光线上的采样点个数
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')

    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 不使用视角数据作为输入
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    # 0 使用位置编码，-1 不使用位置编码
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')

    # L=10，采样点的位置编码维度
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    # L=4，光线的位置编码维度
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # 仅进行渲染不用于训练
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染test数据集
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    # 下采样的倍数
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    # 中心裁剪的训练轮数（对于合成数据集可以优先初始时先对中心区域训练以快速重建，因为合成数据集的照片一般物体处于中心位置附近）
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    # 数据格式
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')

    # 对于大的数据集，test和val数据集，只使用其中的一部分数据，即多个图片中采集一张用于训练，不是所有图片都参与训练
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    # 白色背景，可能会优先将网络表示的空间设置为白色，这样后期训练对于背景空区域直接损失为0加速训练的收敛
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # 使用一半分辨率，对合成数据集一般使用这个参数，因此合成数据集一般分辨率很高可以模糊一点加速网络训练而不影响效果
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    # log输出的频率
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    # 保存模型的频率
    # 每隔1w保存一个
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    # 执行测试集渲染的频率
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    # 执行渲染视频的频率
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser
