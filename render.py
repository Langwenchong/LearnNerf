import torch
import numpy as np
import torch.nn.functional as F

from nerf_helpers import sample_pdf, get_rays, ndc_rays

DEBUG = False

__all__ = ['render', 'batchify_rays', 'render_rays', 'raw2outputs']

# 传入光线，图片信息，模型信息进行模型的采样点训练与体渲染，但是这个render函数主要是完成光线采样训练前的数据预处理


def render(H, W, K,
           chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K:  相机内参 focal
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        # 光线的起始位置, 方向
        rays_o, rays_d = rays

    # 5d输入时，对输入的观察向量（已经转为3d)进行归一化
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # 静态相机 相机坐标到世界坐标的转换
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # 单位向量 [bs,3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    # [bs,1],[bs,1]
    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    # 8=3+3+1+1，虽然光线起始点是rays_o，但是最终计算采样点是从near开始沿着rays_d方向延伸到far
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        # 加了direction的三个坐标
        # 3 3 1 1 3
        rays = torch.cat([rays, viewdirs], -1)  # [bs,11]

    # Render and reshape

    # rgb_map,disp_map,acc_map,raw,rbg0,disp0,acc0,z_std
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        # 对所有的返回值进行reshape
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # 将精细网络的输出单独拿了出来
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    # 前三是list，后5还是在map中
    return ret_list + [ret_dict]

# 如果超过chunk值分批式处理，然后对每一个光线进行采样点训练


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """
    Render rays in smaller minibatches to avoid OOM.
    rays_flat: [N_rand,11]
    """

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    # 将分批处理的结果拼接在一起
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


# 这里面会经过神经网络,这里就是将采样光线和对应的模型放到一起准备开始走网络预测每一个像素颜色
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction. 单位大小查看方向
      粗网络
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.

      raw 是指神经网络的输出
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.


      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.

      精细网络中的光线上的采样频率
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      精细网络
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background. 白色背景
      raw_noise_std: ...


      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]  # N_rand
    # 光线起始位置，光线的方向
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    # 视角的单位向量
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(
        ray_batch[..., 6:8], [-1, 1, 2])  # [bs,1,2] near和far
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    # 采样点，先生成等间隔
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)  # 均匀插值采样
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    # [N_rand,64] -> [N_rand,64]
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples，64个采样点的中点，因此是63个中点
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 中点的附近两个采样点分别是upper和lower
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals,取一个随机值
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        # [bs,64] 加上随机的噪声，然后使用随机的t进行线性插值得到有噪声的采样点，这样就不是均匀采样了但是比均匀采样更加具有鲁棒性
        z_vals = lower + (upper - lower) * t_rand

    # 空间中的采样点
    # [N_rand, 64, 3]
    # 出发点+距离*方向，因此rays_d一开始设置的z值为-1，为了方便这里统一进行与距离z_vals的相乘
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # 使用神经网络 viewdirs [N_rand,3], network_fn 指的是粗糙NeRF或者精细NeRF
    # raw [bs,64,3]，每一条光线采样点累加得到映射的rgbσ
    raw = network_query_fn(pts, viewdirs, network_fn)

    # rgb值，位移图，权重的和，weights就是论文中的那个Ti和alpha的乘积，深度图
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest)

    # 精细网络部分
    if N_importance > 0:
        # _0 是第一个阶段 粗糙网络的结果
        # 这三个留着放在dict中输出用
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # 第二次计算mid，取中点位置63个值
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 注意这里的weights并没有全部传进来，这里主要是参考粗网络计算得到的weights得到一个概率分布重新规划128个采样点的分布位置
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 给精细网络使用的点
        # [N_rays, N_samples + N_importance, 3]，这里应该为192个点，之所以加上粗网络的点是为了添加更多参数值，整体上采样点也是较为合理的
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine

        # 使用神经网络
        # create_nerf 中的 network_query_fn 那个lambda 函数
        # viewdirs 与粗糙网络是相同的
        raw = network_query_fn(pts, viewdirs, run_fn)

        # raw2ouputs是体渲染函数
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

    if retraw:
        # 如果是两个网络，那么这个raw就是最后精细网络的输出
        ret['raw'] = raw

    if N_importance > 0:
        # 下面的0是粗糙网络的输出
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0

        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    # 检查是否有异常值
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        Model的输出
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        采样，并未变成空间点的那个采样点
        z_vals: [num_rays, num_samples along ray]. Integration time.
        光线的方向
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        RGB颜色值
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        逆深度
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        权重和？
        acc_map: [num_rays]. Sum of weights along each ray.
        权重
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        估计的深度
        depth_map: [num_rays]. Estimated distance to object.
    """
    # Alpha的计算
    # relu：负数拉平为0，raw就是算出来的σ,dists是采样点距离差
    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw) * dists)
    # [bs,63]
    # 采样点之间的距离，因为加入了噪声因此未必是均匀的等值距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # todo 不知道在干嘛，总之就是对于每一个光线的一组采样点之差后面再拼接一下1e10,这里是为了模拟无限远处，最后一个邻域无限远因此增幅较强，可以认为基本只有距离衰减
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    # rays_d[...,None,:] [bs,3] -> [bs,1,3]
    # 1维 -> 3维
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    # RGB经过sigmoid处理激活保证都是正值，softmax与sigmoid的区别是最终值归一化保证求得的是权重概率
    # 注意这里的rgb仅仅是一个采样点的颜色值
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    # 计算公式3 [bs, 64],
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # 后面这部分就是Ti，前面是alpha，这个就是论文上的那个权重w [bs,64]，这里进行了一点点变形，发现Ti可以用1-a的累乘间接表示
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)),
                                               1. - alpha + 1e-10], -1),
                                    -1)[:, :-1]
    # [bs, 64,1] * [bs,64,3]
    # 在第二个维度，64将所有的点的值相加 -> [32,3]
    # 公式3的结果值，这里才是每一个像素即累加像素点后计算得出的颜色值
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    # (32,)
    # 深度图
    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, -1)
    # 视差图
    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                              depth_map / torch.sum(weights, -1))

    # 权重和
    # 这个值仅做了输出用，后续并无使用
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map
