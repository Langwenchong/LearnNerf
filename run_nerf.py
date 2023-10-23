import os
import imageio
import time
import torch
import numpy as np
from tqdm import tqdm, trange
from nerf_helpers import *
from nerf_model import NeRF
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from render import *
from inference import render_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        # 以chunk分批进入网络，防止显存爆掉，然后在拼接
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    被下面的create_nerf 封装到了lambda方法里面
    Prepares inputs and applies network 'fn'.
    inputs: pts，光线上的点 如 [1024,64,3]，1024条光线，一条光线上64个点
    viewdirs: 光线起点的方向
    fn: 神经网络模型 粗糙网络或者精细网络
    embed_fn:
    embeddirs_fn:
    netchunk:
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [N_rand*64,3]
    # 坐标点进行编码嵌入 [N_rand*64,63]
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 方向进行位置编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # [N_rand*64,27]
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 里面经过网络 [bs*64,3]
    outputs_flat = batchify(fn, netchunk)(embedded)
    # [bs*4,4] -> [bs,64,4]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    # 注意他不是定义网络，而是将参数进一步解析首先创建粗网络与惊喜网络，之后将网络组合，以及后面训练需要用到的优化器返还，而且要注意他还返还训练开关run_network这个lambda函数！！！
    # 具体的训练前输入的数据在主循环实现,如从像素点引出射线,采样点设置等
    """Instantiate NeRF's MLP model.
    """

    # 对于点进行位置编码，L=10
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # 对光线视角Θφ进行编码，L=4
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    # 想要=5生效，首先需要use_viewdirs=False and N_importance>0
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    # 创建粗网络
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # 创建精细网络
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        # 模型参数
        grad_vars += list(model_fine.parameters())

    # input是xyz，viewdir是方向向量，network_fn是上面定义的粗网络或者惊喜网络 
    # 这个run_network才是真正执行粗或者精细网络的过程，而粗网络与网络这个整体的coarse2fine的架构叫做NeRF模型
    # 这里的run_network之所以还要接受一个netchunk实际上才是真正一次性网络处理的点的个数，由此实现了当光线很多或者采样点过多时代码也会自行进一步切割合理的将数据传入以防显存爆满
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    # 优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # 开始的轮次
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints，因此可以中途开始
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)

    # load参数
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        # 精细网络
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        # 粗网络
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    print(model_fine)

    # NDC only good for LLFF-style forward facing data，标准设备坐标仅适用于前景，对于全景需要参考nerf++或者mip nerf 360
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # 将这个模型的训练参数，测试参数，开始的轮次，训练的参数，优化器返还
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# ----------------------------------------------------------------------------------------------------------------------

def create_log_files(basedir, expname, args):
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # 保存一份参数文件
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # 保存一份配置文件
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    return basedir, expname


def run_render_only(args, images, i_test, basedir, expname, render_poses, hwf, K, render_kwargs_test, start):
    with torch.no_grad():
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                              savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)


def train():
    # 解析参数
    from opts import config_parser
    parser = config_parser()
    args = parser.parse_args()

    # --------------------------------------------------------------------------------------------------------

    # Load data

    # 在这个数据集会特殊些 LINEMOD
    K = None

    # 一共有四种类型的数据集
    # 是configs目录中 只有llff和blender两种类型
    # 原始的nerf仓库中有deepvoxels类型的数据
    # LINEMOD 没见过

    # llff Local Light Field Fusion
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # images，所有的图片，train val test在一起，poses也一样
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # 初始化采样点在空间中的位置，这个near与far需要将物体包裹住，超参数需要合理赋值
        near = 2.
        far = 6.

        if args.white_bkgd:
            # todo 这个是什么操作，为什么白色背景要这样操作，推测是可以防止模型取重建拟合白色背景
            #  根据Alpha混合的原理，对于每个像素，将RGB通道乘以Alpha通道的值，然后加上（1-Alpha)通道的值，这段代码的作用是将输入的图像进行Alpha混合操作，输出只保留RGB通道，并根据Alpha通道的值进行混合计算
            # 最终的输出图像将只包含RGB通道，且根据Alpha通道的值进行了混合处理，达到了透明度混合的效果
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        # 这个数据类型 原始的nerf中没有

        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    # 高宽之前都是浮点数类型，需要进行量化截断
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # K 相机内参 focal 是焦距，0.5w 0.5h 是中心点坐标
    # 这个矩阵是相机坐标到图像坐标转换使用
    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # --------------------------------------------------------------------------------------------------------

    # render the test set instead of render_poses path
    # 使用测试集的pose，而不是用那个固定生成的render_poses(40帧360°视频)
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # --------------------------------------------------------------------------------------------------------

    # Create log dir and copy the config file

    basedir = args.basedir
    expname = args.expname

    # 生成日志函数
    create_log_files(basedir, expname, args)

    # --------------------------------------------------------------------------------------------------------

    # Create nerf model
    # 创建好整个coarse2fine的NeRF整体模型架构,获得训练测试的参数,起始轮次,优化器,以及训练的开关run_network函数
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    # 有可能从中间迭代恢复运行的
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # --------------------------------------------------------------------------------------------------------

    # Short circuit if only rendering out from trained model
    # 这里会使用render_poses
    if args.render_only:
        # 仅进行渲染，不进行训练
        print('RENDER ONLY')
        run_render_only(args, images, i_test, basedir, expname, render_poses, hwf, K, render_kwargs_test, start)
        return

    # --------------------------------------------------------------------------------------------------------

    # Prepare ray batch tensor if batching random rays
    N_rand = args.N_rand

    use_batching = not args.no_batching
    # 使用的话就是打混图片从中选取1024个训练,也就是并不是所有图片都参与训练,一般合成数据集这样处理
    # 对于前景等则一般所有图片各取1024个点训练
    # 注意即使训练的时候也不是对图片的所有点进行训练,这里的use_batching的区别是1024是针对于单个图片还是所有图片
    # 在真正渲染时都是对某个特定新视角的所有像素进行渲染
    if use_batching:
        # For random ray batching
        print('get rays')  # (img_count,2,400,400,3) 2是 rays_o和rays_d
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')  # rays和图像混在一起
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only, 仅使用训练文件夹下的数据
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)  # 打乱光线

        print('done')
        i_batch = 0

    # 统一一个时刻放入cuda
    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    poses = torch.Tensor(poses).to(device)

    # --------------------------------------------------------------------------------------------------------

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # 训练部分的代码
    # 两万次迭代
    # 可能是强迫症，不想在保存文件的时候，出现19999这种数字
    N_iters = 200000 + 1
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            # 一批光线
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]

            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]  # 前两个是rays_o和rays_d, 第三个是target就是image的rgb

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # 所用光线用过之后，重新打乱
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]  # [400,400,3] 图像内容
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                # precrop_iters: number of steps to train on central crops
                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    # coords = torch.stack(
                    #     torch.meshgrid(
                    #         torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    #         torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW), indexing='ij',
                    #     ), -1)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW), 
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H),
                                                        torch.linspace(0, W - 1, W), indexing='ij'),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                # 选出的像素坐标
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)  # 堆叠 o和d
                # target 也同样选出对应位置的点
                # target 用来最后的mse loss 计算
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # rgb 网络计算出的图像
        # 前三是精细网络的输出内容，其他的还保存在一个dict中，有5项
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)
        #  最终使用精细网络输出的体渲染像素颜色值与gt求loss方向传播更新参数
        optimizer.zero_grad()
        # 计算loss
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        # 计算指标
        psnr = mse2psnr(img_loss)

        # rgb0 粗网络的输出
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # 学习率衰减
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # 保存模型
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                # 运行的轮次数目
                'global_step': global_step,
                # 粗网络的权重
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 精细网络的权重
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                # 优化器的状态
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # 生成测试视频，使用的是render_poses (这个不等同于test数据)
        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # 360度转一圈的视频
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # 执行测试，使用测试数据
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        # 用时
        dt = time.time() - time0
        # 打印log信息的频率
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Time: {dt}")

        global_step += 1


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
