import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import numbers, math

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_3 = self.dwconv3(x)
        x1, x2 = x_3.chunk(2, dim=1)
        x = F.gelu(x1) * x2 + F.gelu(x2) * x1
        x = self.project_out(x)
        return x


class SpatialMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bimamba_type="none", if_devide_out=False):
        super().__init__()

        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(d_inner, dtype=torch.float32))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        if bimamba_type != "none":
            self.conv1d_b = nn.Conv1d(
                in_channels=d_inner,
                out_channels=d_inner,
                bias=True,
                kernel_size=d_conv,
                groups=d_inner,
                padding=d_conv - 1,
            )
            self.x_proj_b = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            self.dt_proj_b = nn.Linear(dt_rank, d_inner, bias=True)

            A_b = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
            self.A_b_log = nn.Parameter(torch.log(A_b.float()))
            self.D_b = nn.Parameter(torch.ones(d_inner, dtype=torch.float32))


    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)
        
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        
        y = self.ssm(x, res)
        
        if self.bimamba_type != "none":
            x_b = rearrange(x, 'b l d_in -> b d_in l')
            x_b = self.conv1d_b(x_b)[:, :, :l]
            x_b = rearrange(x_b, 'b d_in l -> b l d_in')
            x_b = F.silu(x_b)
            y_b = self.ssm_b(x_b, res)

            if not self.if_devide_out:
                y =  y + y_b
            else:
                y = (y + y_b) / 2

        output = self.out_proj(y)
        return output

    def ssm(self, x, res):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log)
        D = self.D

        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        y = self.selective_scan(x, delta, A, B, C, D)
        y = y * F.silu(res)
        return y

    def ssm_b(self, x, res):
        (d_in, n) = self.A_b_log.shape
        A_b = -torch.exp(self.A_b_log)
        D_b = self.D_b

        x_dbl = self.x_proj_b(x)
        delta, B, C = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj_b(delta))
        
        y = self.selective_scan(x, delta, A_b, B, C, D_b, reverse=True)
        y = y * F.silu(res)
        return y

    def selective_scan(self, u, dt, A, B, C, D, reverse=False):
        dA = torch.einsum('bld,dn->bldn', dt, A)
        dB_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)

        if reverse:
            dA = dA.flip(1)
            dB_u = dB_u.flip(1)

        dA_cumsum = F.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
        x = dB_u * dA_cumsum
        x = x.cumsum(1) / (dA_cumsum + 1e-12)
        y = torch.einsum('bldn,bln->bld', x, C)

        if reverse:
            y = y.flip(1)

        return y + u * D



class MultiScalePooling(nn.Module):
    def __init__(self, dim, scales):
        super(MultiScalePooling, self).__init__()
        self.msp_branches = nn.ModuleList([nn.AdaptiveAvgPool2d((scale, scale)) for scale in scales])
        self.conv_branches = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for _ in scales])
        self.scales = scales

    def forward(self, feature):
        bs, ch, h, w = feature.size()
        pool_feas = []
        for i, layer in enumerate(self.msp_branches):
            pooled = layer(feature)
            conv = self.conv_branches[i](pooled)
            pool_feas.append(conv.view(bs, -1, conv.size(2) * conv.size(3)))
        output = torch.cat(pool_feas, dim=2)
        return output, (h, w)

class MultiScalePoolingReverse(nn.Module):
    def __init__(self, dim, scales):
        super(MultiScalePoolingReverse, self).__init__()
        self.scales = scales
        self.dim = dim
        self.n = len(scales)
        self.conv = nn.Conv2d(dim * self.n,  dim, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(dim * self.n, dim * self.n, kernel_size=3, stride=1, padding=1, groups=dim * self.n, bias=False)
    def forward(self, pooled_feature, original_size):
        bs, ch, n = pooled_feature.size()
        H, W = original_size  # Use the provided original size
        scale_features = []
        idx = 0
        for scale in self.scales:
            size = scale * scale
            scale_feat = pooled_feature[:, :, idx:idx + size]
            scale_feat = scale_feat.view(bs, ch, scale, scale)
            scale_feat = F.interpolate(scale_feat, size=(H, W), mode='bilinear', align_corners=False)
            scale_features.append(scale_feat)
            idx += size
        output = torch.cat(scale_features, dim=1)
        output = self.conv(self.dwconv(output))
        return output

class PyramidpoolingMamba(nn.Module):
    def __init__(self, dim, scales, depth_p):
        super(PyramidpoolingMamba, self).__init__()
        self.mp_f = MultiScalePooling(dim, scales)
        self.mp_r = MultiScalePoolingReverse(dim, scales)
        self.conv = nn.Conv2d(dim,  dim, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.model1 = nn.Sequential(*[SpatialMambaBlock(d_model=dim) for i in range(depth_p)])
        
    def forward(self, x0):
        res = x0
        b, c, h, w = x0.shape
        x = self.dwconv(self.conv(x0))
        x, original_size  = self.mp_f(x)
        x_mamba = self.model1(x.permute(0, 2, 1).contiguous())
        out1 = self.mp_r(x_mamba.permute(0, 2, 1).contiguous(), original_size) 
        return out1 



class AxispoolingMamba(nn.Module):
    def __init__(self, dim, depth_a):
        super(AxispoolingMamba, self).__init__()
        self.ln_1 = LayerNorm(dim, LayerNorm_type='WithBias')
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.model1 = nn.Sequential(*[SpatialMambaBlock(d_model=dim) for i in range(depth_a)])
        
    def forward(self, x0):
        res = x0
        b, c, h, w = x0.shape
        x = x0
        
        x_h = self.pool_h(x).view(b, c, h)  # Changed from -1 to h for compatibility
        x_mamba_h = self.model1(x_h.permute(0, 2, 1).contiguous())
        x_mamba_h = x_mamba_h.permute(0, 2, 1).contiguous().view(b, c, h, 1)  # Correct shape
        
        x_mamba_h = x_mamba_h * x
        
        x_w = self.pool_w(x_mamba_h).view(b, c, w)  # Changed from -1 to w for compatibility
        x_mamba_w = self.model1(x_w.permute(0, 2, 1).contiguous())
        x_mamba_w = x_mamba_w.permute(0, 2, 1).contiguous().view(b, c, 1, w)  # Correct shape
        
        x_mamba_w = x_mamba_w * x
        
        return x_mamba_w

class DualpoolingMamba(nn.Module):
    def __init__(self, dim, scales, depth_p, depth_a):
        super(DualpoolingMamba, self).__init__()
        self.ln_1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.Global = PyramidpoolingMamba(dim, scales, depth_p)
        self.Local = AxispoolingMamba(dim, depth_a)
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln_2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.mlp = FeedForward(dim)
    def forward(self, x0):
        x = self.ln_1(x0)

        x_g = self.Global(x) 
        x_l = self.Local(x) 
        x_gl = self.conv(torch.cat([x_g, x_l], dim=1)) + x0

        out = self.mlp(self.ln_2(x_gl)) + x_gl
        return out


class feature_pyramid_encoder(nn.Module):
    def __init__(self, channels):
        super(feature_pyramid_encoder, self).__init__()
        self.block0 = DualpoolingMamba(channels, scales=[12, 16, 20, 24, 28], depth_p=2, depth_a=1) 

        self.down0 = Downsample(channels)

        self.block1 = DualpoolingMamba(channels * 2, scales=[6, 8, 10, 12, 14], depth_p=2, depth_a=1) 

        self.down1 = Downsample(channels * 2)

        self.block2 = DualpoolingMamba(channels * 4, scales=[3, 4, 5, 6, 7], depth_p=2, depth_a=1) 

        self.down2 = Downsample(channels * 4)

        self.block3 = DualpoolingMamba(channels * 8,  scales=[1, 2, 3, 4, 5], depth_p=4, depth_a=2)

    def forward(self, x):
        pyramid = []
        level0 = self.block0(x)
        pyramid.append(level0)
        level1 = self.block1(self.down0(level0))
        pyramid.append(level1)
        level2 = self.block2(self.down1(level1))
        pyramid.append(level2)
        level3 = self.block3(self.down2(level2))
        pyramid.append(level3)
        return pyramid


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class feature_pyramid_decoder(nn.Module):
    def __init__(self, channels):
        super(feature_pyramid_decoder, self).__init__()

        self.up3 = Upsample(channels * 8)

        self.block3 = DualpoolingMamba(channels * 4, scales=[3, 4, 5], depth_p=2, depth_a=1) 

        self.up2 = Upsample(channels * 4)

        self.block2 = DualpoolingMamba(channels * 2, scales=[6, 8, 10], depth_p=2, depth_a=1)

        self.up1 = Upsample(channels * 2)

        self.block1 = DualpoolingMamba(channels, scales=[12, 16, 20], depth_p=2, depth_a=1)  

    def forward(self, x4, x3, x2, x1):
        x3 = self.block3(self.up3(x4) + x3)  
        x2 = self.block2(self.up2(x3) + x2)
        x1 = self.block1(self.up1(x2) + x1)
        return x1


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=2)

        self.encoder = feature_pyramid_encoder(out_channels)
        self.decoder = feature_pyramid_decoder(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out0 = self.conv1(self.relu(self.conv0(x)))

        pyramid_fea = self.encoder(out0)

        level4, level3, level2, level1 = pyramid_fea[3], pyramid_fea[2], pyramid_fea[1], pyramid_fea[0]

        out = self.decoder(level4, level3, level2, level1)
        out1 = self.relu(self.conv2(out))
        final_out = self.relu(self.conv3(out1)) + residual

        return final_out







    
if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = Net(in_channels=3, out_channels=48).cuda()
    print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    model(inputs)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')