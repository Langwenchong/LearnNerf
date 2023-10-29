import torch
import torch.nn as nn
import numpy as np

__all__ = ['img2mse', 'mse2psnr', 'to8b', 'get_embedder',
           'get_rays', 'get_rays_np', 'ndc_rays', 'sample_pdf']

# Misc


def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']  # 3
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # sin(x),sin(2x),sin(4x),sin(8x),sin(16x),sin(32x),sin(64x),sin(128x),sin(256x),sin(512x)
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns

        # 3D坐标是63，2D方向是27
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# 位置编码相关
def get_embedder(multires, i=0):
    """
    multires: 3D 坐标是10，2D方向是4
    """
    if i == -1:
        # 直接使用xyz，效果会不好
        return nn.Identity(), 3

    embed_kwargs = {
        # 编码后是否将初始输入也拼接进来，比如对于xyz编码后是60维，如果维True，还会加上xyz本身，因此一个点的输入维度维63维
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    # 第一个返回值是lamda，给定x，返回其位置编码以及返还的新的编码向量的维度
    return embed, embedder_obj.out_dim


# ----------------------------------------------------------------------------------------------------------------------

# Ray helpers
def get_rays(H, W, K, c2w):
    """
    K：相机内参矩阵
    c2w: 相机到世界坐标系的转换
    """
    # i
    # [0,......]
    # [1,......]
    # [W-1,....]
    # j
    # [0,..,H-1]
    # [0,..,H-1]
    # [0,..,H-1]

    # 新版本pytorch.meshgrid取消了参数indexing，但是默认还是ij，这里是对像素平面进行坐标划分
    # i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))
    # 注意这里转置了，所以
    # j
    # [0,......]
    # [1,......]
    # [H-1,....]
    # i
    # [0,..,w-1]
    # [0,..,w-1]
    # [0,..,w-1]
    # 可以理解为以左上角为原点的像素平面坐标系，横轴坐标对应i，纵轴坐标对应y
    i = i.t()
    j = j.t()
    # [400,400,3]，最终得到每一个相机到像素的方向向量（未必归一化），并且此时是在camera坐标系中
    # 默认z都是-1这样保证方向都是同一个平面，同时在后面与zvals相乘能够保证量化距离，并且这里得到是以摄像机为原点的的相机坐标系下的向量坐标
    # 这里之所以要除以焦距，是为了得到深度为-1时光线向量的x和y应该是多少，进一步与zvals相乘就可以量化表示任意深度下的逆深度向量了(注意这里只是像平面->相机坐标系)
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dirs [400,400,3] -> [400,400,1,3]
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d [400,400,3]
    # 转为世界坐标系，但是还没有归一化为单位向量
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 前三行，最后一列，定义了相机的平移(回忆相机外参是[R|t])，因此可以得到射线的原点o其实就是相机在世界坐标系下的坐标
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    # 与上面的方法相似，这个是使用的numpy，上面是使用的torch
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -
                    (j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    bins: z_vals_mid
    """

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    # 归一化 [bs, 62]，由于默认设定采取光线1024，所以这里其实batchsize就是1024
    # 概率密度函数
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 累积概率分布函数，

    # torch.cumsum的使用方法，其实就是逐个累加
    # tensor_res = torch.arange(0, 6).view(2, 3)
    # 输出：
    # tensor_res = tensor([[0, 1, 2],
    #                      [3, 4, 5]])
    # aaaa = tensor_res.cumsum(dim=-1)
    # 输出：
    # aaaa = tensor([[0, 1, 3],
    #                [3, 7, 12]])

    cdf = torch.cumsum(pdf, -1)
    # 在第一个位置补0
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # 按照这个新的分布函数随机采样128个采样点的概率值
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])  # [bs,128]

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF

    u = u.contiguous()
    # u 是随机生成的概率值，符合原64个点的分布函数规律
    # 128个新的采样点值找到对应的插入的位置(是索引位置)，注意cdf是概率分布，因此介于[0,1],u也是
    inds = torch.searchsorted(cdf, u, right=True)
    # 这里202~207主要是为了保证新的128个采样点坐标一定在之前的64个采样点中间，如果超出范围强制界定在范围内部，自此每一个新插入点都找到了要插入在原64个采样点的前后的索引位置
    # 前一个位置，为了防止inds中的0的前一个是-1，这里就还是0，即要插入到64个点中的那个位置，当同级别时默认原位置元素整体后移
    # 因此插入到最前面只需要为0
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 最大的位置就是cdf的上限位置，防止过头，跟上面的意义相同
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # (batch, N_samples, 2)，这里记录了每一个新采样点相邻的原64采样点的位置，因此会有许多小区间相同因为大部分新采样点会在密度较大的两个原采样点分段之间分布
    inds_g = torch.stack([below, above], -1)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # (batch, N_samples, 63)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # 如[1024,128,63] 提取 根据 inds_g[i][j][0] inds_g[i][j][1]
    # cdf_g [1024,128,2]，这里和inds_g的区别是，inds_g记录的是新的采样点前后相邻的原64点的索引位置，这里是其对应的概率值
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 如上, bins 是从2到6的采样点，是64个点的中间值，这里是拿到新采样点相邻的原63采样点中心值对应的索引位置
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 差值计算的是原先64个采样点每两个点之间分段的权重差值（注意不是插值），仅是简单求差
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    # 防止过小
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # 这里就是线性插值了，即计算的是
    # 新的采样点u概率值-与他相邻的前面最近的原64点的概率值/与新采样点相邻的两个原64采样点的概率差值得到了插值比例系数
    t = (u - cdf_g[..., 0]) / denom

    # lower+线性插值得到128个采样更加合理的点
    # 因此这里是使用63个中点还是插值得到，只不过如果某个地方是物体表面附近，那么会有多个t值接近因此就会使用更多次接近物体表面的中点来做插值生成真正的新的128个采样点的坐标深度值
    # 总之一句话：真正的采样点坐标值都是使用中点距离插值+lower得到，前面做的u，inds_g,bins_g,cdf_g都是在使用概率分布随机生成概率来计算最终新的采样点的插值比例因子
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# ----------------------------------------------------------------------------------------------------------------------

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * \
        (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * \
        (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
