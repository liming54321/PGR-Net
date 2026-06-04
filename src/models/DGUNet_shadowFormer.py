import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numpy as np
import cv2
from collections import OrderedDict
from torchvision.ops import deform_conv2d
from .builder import MODEL_REGISTRY
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class SKConv(nn.Module):
    def __init__(self, features, WH=64, M=2, G=8, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3 + 2 * i, stride=stride, padding=1 + i, groups=G),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True)
                )
            )
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        feats = [conv(x).unsqueeze(dim=1) for conv in self.convs]  # shape: [B, M, C, H, W]
        feats = torch.cat(feats, dim=1)                            # shape: [B, M, C, H, W]
        U = feats.sum(dim=1)                                       # shape: [B, C, H, W]
        s = F.adaptive_avg_pool2d(U, 1).view(U.size(0), -1)        # shape: [B, C]
        z = self.fc(s)                                             # shape: [B, d]
        weights = [fc(z).unsqueeze(dim=1) for fc in self.fcs]      # [B, 1, C] * M
        attention_weights = torch.cat(weights, dim=1)              # shape: [B, M, C]
        attention_weights = self.softmax(attention_weights)       # softmax over M
        V = (feats * attention_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # shape: [B, C, H, W]
        return V

class MixedResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False, stride=1):
        super(MixedResidualBlock, self).__init__()
        self.dw_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_ch)
        )
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=bias, stride=stride) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dw_path(x) + self.conv_path(x) + self.shortcut(x)
        return self.relu(out)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias
    )
def conv_down(in_chn, out_chn, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)


def default_conv(input_ch, output_ch, kernel_size, bias=True):
    return nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size,
                     padding=kernel_size // 2, bias=bias)

def conv2(in_chn, out_chn, kernel_size, padding, bias):
    return nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size,
                     padding=padding, bias=bias)

def conv3(in_chn, out_chn, kernel_size, stride, bias):
    return nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size,
                     padding=kernel_size // 2, bias=bias, stride=stride)

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.conv1(concat)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias)
        ]
        self.body = nn.Sequential(*modules_body)
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.SA = SpatialAttention(kernel_size=7)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.SA(res)
        res += x
        return res

class SAM(nn.Module):  # DGUNet
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img

class mergeblock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias)

    def forward(self, x, bridge):
        out = torch.cat([x, bridge], 1)
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)
        V_t = sub.view(b_, self.num_subspace, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(dim=2).unsqueeze(-1))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_ * w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out + x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.in_features * self.hidden_features
        # fc2
        flops += H * W * self.hidden_features * self.out_features
        print("MLP:{%.2f}" % (flops / 1e9))
        return flops

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 3
        return flops

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., se_layer=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.ll = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xm, attn_kv=None, mask=None):
        B_, N, C = x.shape
        one = torch.ones_like(xm)
        zero = torch.zeros_like(xm)
        xm = torch.where(xm < 0.1, one, one * 2)
        mm = xm @ xm.transpose(-2, -1)
        one = torch.ones_like(mm)
        mm = torch.where(mm == 2, one, one * 0.2)
        mm = torch.unsqueeze(mm, dim=1)
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) * mm

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.ll(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        flops += self.qkv.flops(H, W)
        if self.token_projection != 'linear_concat':
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
            flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        else:
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N * 2
            flops += nW * self.num_heads * N * N * 2 * (self.dim // self.num_heads)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            act_layer()
        )
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128, 128)):
        bs, hw, c = x.size()
        hh, ww = img_size
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=ww)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=ww)
        x = self.linear2(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.dim * self.hidden_dim  # linear1
        flops += H * W * self.hidden_dim * 3 * 3     # DWConv
        flops += H * W * self.hidden_dim * self.hidden_dim  # PWConv
        flops += H * W * self.hidden_dim * self.dim  # linear2
        print("LeFF:{%.2f}" % (flops / 1e9))
        return flops

class SIMTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=10, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection, se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.CAB = CAB(dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, xm, mask=None, img_size=(128, 128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        assert L == W * H, \
            f"Input image size ({H}*{W} doesn't match model ({L})."

        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W), mode='nearest').permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        if self.shift_size > 0:
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        xm = xm.permute(0, 2, 3, 1)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_m = torch.roll(xm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_m = xm

        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        m_windows = window_partition(shifted_m, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        m_windows = m_windows.view(-1, self.win_size * self.win_size, 1)  # nW*B, win_size*win_size, C
        attn_windows = self.attn(x_windows, m_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)

        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        x = self.CAB(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), img_size=img_size))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        print("LeWin:{%.2f}" % (flops / 1e9))
        return flops

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, depth=5):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()  # []
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(
                UNetConvBlock(in_size=n_feat + scale_unetfeats * i, out_size=n_feat + scale_unetfeats * (i + 1),
                              downsample=True, relu_slope=0.2, use_csff=csff, use_HIN=True))
        self.body.append(UNetConvBlock(in_size=n_feat + scale_unetfeats * (depth - 1),
                                       out_size=n_feat + scale_unetfeats * (depth - 1), downsample=False,
                                       relu_slope=0.2, use_csff=csff, use_HIN=True))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x, encoder_outs[i], decoder_outs[-i - 1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = conv2(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = conv2(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out * torch.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(out_size * 2, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=5):
        super(Decoder, self).__init__()

        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()  # []
        for i in range(depth - 1):
            self.body.append(UNetUpBlock(in_size=n_feat + scale_unetfeats * (depth - i - 1),
                                         out_size=n_feat + scale_unetfeats * (depth - i - 2), relu_slope=0.2))
            self.skip_conv.append(
                conv3(n_feat + scale_unetfeats * (depth - i - 1), n_feat + scale_unetfeats * (depth - i - 2), 3, 1, 1))

    def forward(self, x, bridges):
        res = []
        for i, up in enumerate(self.body):
            x = up(x, self.skip_conv[i](bridges[-i - 1]))
            res.append(x)

        return res

class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

@MODEL_REGISTRY.register()
class DGUNet_shadowFormer(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=32,
                 scale_unetfeats=20, kernel_size=3, reduction=4,
                 bias=False, depth=5):
        super(DGUNet_shadowFormer, self).__init__()
        act = nn.PReLU()

        # ---------- Stage 1 Modules ----------
        self.shallow_feat1 = nn.Sequential(
            MixedResidualBlock(3, n_feat), SKConv(n_feat),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.phi_0   = ResBlock(default_conv, 3, 3)
        self.phit_0  = ResBlock(default_conv, 3, 3)
        self.r0      = nn.Parameter(torch.Tensor([0.5]))
        self.stage1_encoder = Encoder(n_feat * 3, kernel_size, reduction,
                                      act, bias, scale_unetfeats,
                                      depth=4, csff=False)
        self.stage1_decoder = Decoder(n_feat * 3, kernel_size, reduction,
                                      act, bias, scale_unetfeats,
                                      depth=4)
        self.sim_blcok_0 = SIMTransformerBlock(
            dim=156,
            input_resolution=(32, 32),
            num_heads=12,
            win_size=8
        )

        # ---------- Stage 2 Modules ----------
        self.shallow_feat2 = nn.Sequential(
            MixedResidualBlock(3, n_feat), SKConv(n_feat),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.phi_1   = ResBlock(default_conv, 3, 3)
        self.phit_1  = ResBlock(default_conv, 3, 3)
        self.r1      = nn.Parameter(torch.Tensor([0.5]))
        self.stage2_encoder = Encoder(n_feat*3, kernel_size, reduction,
                                      act, bias, scale_unetfeats,
                                      depth=4, csff=True)
        self.stage2_decoder = Decoder(n_feat*3, kernel_size, reduction,
                                      act, bias, scale_unetfeats,
                                      depth=4)
        self.sim_blcok_1    = SIMTransformerBlock(dim=156,
                                input_resolution=(32,32),
                                num_heads=12, win_size=8)

        # ---------- Stage 3 Modules ----------
        self.shallow_feat6 = nn.Sequential(
            MixedResidualBlock(3, n_feat), SKConv(n_feat),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.phi_5   = ResBlock(default_conv, 3, 3)
        self.phit_5  = ResBlock(default_conv, 3, 3)
        self.r5      = nn.Parameter(torch.Tensor([0.5]))
        self.stage6_encoder = Encoder(n_feat*3, kernel_size, reduction,
                                      act, bias, scale_unetfeats,
                                      depth=4, csff=True)
        self.stage6_decoder = Decoder(n_feat*3, kernel_size, reduction,
                                      act, bias, scale_unetfeats,
                                      depth=4)
        self.sim_blcok_6    = SIMTransformerBlock(dim=156,
                                input_resolution=(32,32),
                                num_heads=12, win_size=8)

        # ---------- Stage 4 Modules ----------
        self.shallow_feat7 = nn.Sequential(
            MixedResidualBlock(3, n_feat), SKConv(n_feat),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.phi_6   = ResBlock(default_conv, 3, 3)
        self.phit_6  = ResBlock(default_conv, 3, 3)
        self.r6      = nn.Parameter(torch.Tensor([0.5]))
        self.sim_blcok_7    = SIMTransformerBlock(dim=156,
                                input_resolution=(32,32),
                                num_heads=12, win_size=8)
        self.calayer2 = CALayer(156, reduction=16)
        self.sam12   = SAM(n_feat*3, kernel_size=1, bias=bias)
        self.merge12 = mergeblock(n_feat*3, 3, True)   # Stage1→Stage2
        self.sam23   = SAM(n_feat*3, kernel_size=1, bias=bias)
        self.merge56 = mergeblock(n_feat*3, 3, True)   # Stage2→Stage6
        self.sam67   = SAM(n_feat*3, kernel_size=1, bias=bias)
        self.merge67 = mergeblock(n_feat*3, 3, True)   # Stage6→Stage7

        self.tail    = conv(n_feat*3, out_c, kernel_size, bias=bias)

    def forward(self, img, mask):
        # ---------- Stage 1 ----------
        phixsy_1 = self.phi_0(img) - img
        x1_img   = img - self.r0 * self.phit_0(phixsy_1)
        x1       = self.shallow_feat1(x1_img)
        m1 = F.interpolate(mask, size=x1.shape[2:], mode='nearest')
        m1 = m1.repeat(1, 32, 1, 1)
        x1       = torch.cat([x1, x1*m1, x1*(1-m1)], dim=1)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        f1 = feat_fin1.flatten(2).transpose(1,2).contiguous()
        xm = mask
        for _ in range(3): xm = F.max_pool2d(xm, 2)
        f1 = self.sim_blcok_0(f1, xm, img_size=(32,32))
        f1 = rearrange(f1, 'b (h w) c -> b c h w', h=32, w=32)
        res1 = self.stage1_decoder(f1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        # ---------- Stage 2 ----------
        phixsy_2 = self.phi_1(stage1_img) - stage1_img
        x2_img = stage1_img - self.r1 * self.phit_1(phixsy_2)
        x2 = self.shallow_feat2(x2_img)
        m2 = F.interpolate(mask, size=x2.shape[2:], mode='nearest')
        m2 = m2.repeat(1, 32, 1, 1)
        x2 = torch.cat([x2, x2 * m2, x2 * (1 - m2)], dim=1)
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2, feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        f2 = feat_fin2.flatten(2).transpose(1, 2).contiguous()
        f2 = self.sim_blcok_1(f2, xm, img_size=(32, 32))
        f2 = rearrange(f2, 'b (h w) c -> b c h w', h=32, w=32)
        f2 = self.calayer2(f2)
        res2 = self.stage2_decoder(f2, feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)

        # ---------- Stage 3 ----------
        phixsy_6 = self.phi_5(stage2_img) - stage2_img
        x6_img   = stage2_img - self.r5 * self.phit_5(phixsy_6)
        x6       = self.shallow_feat6(x6_img)
        m6 = F.interpolate(mask, size=x6.shape[2:], mode='nearest')
        m6 = m6.repeat(1, 32, 1, 1)
        x6       = torch.cat([x6, x6*m6, x6*(1-m6)], dim=1)
        x6_cat   = self.merge56(x6, x3_samfeats)
        feat6, feat_fin6 = self.stage6_encoder(x6_cat, feat2, res2)
        f6 = feat_fin6.flatten(2).transpose(1,2).contiguous()
        f6 = self.sim_blcok_6(f6, xm, img_size=(32,32))
        f6 = rearrange(f6, 'b (h w) c -> b c h w', h=32, w=32)
        res6 = self.stage6_decoder(f6, feat6)
        x7_samfeats, stage6_img = self.sam67(res6[-1], x6_img)

        # ---------- Stage 4 ----------
        phixsy_7 = self.phi_6(stage6_img) - stage6_img
        x7_img   = stage6_img - self.r6 * self.phit_6(phixsy_7)
        x7       = self.shallow_feat7(x7_img)
        m7 = F.interpolate(mask, size=x7.shape[2:], mode='nearest')
        m7 = m7.repeat(1, 32, 1, 1)
        x7       = torch.cat([x7, x7*m7, x7*(1-m7)], dim=1)
        x7_cat   = self.merge67(x7, x7_samfeats)
        stage7_img = self.tail(x7_cat) + img

        return [stage7_img, stage6_img, stage2_img, stage1_img]
