"""
FDConv —— Frequency Dynamic Convolution for Dense Image Prediction (CVPR 2025).

本文件是在官方 FDConv_detection/mmdet_custom/FDConv.py 基础上针对 Mamba-YOLO
场景做了如下适配:

1. 移除 mmcv.cnn.CONV_LAYERS 依赖,使其不依赖 mmdet/mmcv 生态。
2. 修复原版在 groups > 1 (包含 depth-wise conv) 时的 reshape 错误,
   所有与"卷积核的输入通道维"相关的张量统一按 in_channels // groups
   构建。
3. 修复 use_fdconv_if_c_gt 判断逻辑,使用 in_channels_per_group 替代总通道数。
4. FBM spatial_group 自适应: groups=1 时自动设为 in_channels,提升逐通道频带调制能力。
5. AMP 兼容: KSM 注意力计算保持与模型权重一致的 dtype,避免验证阶段 dtype 不匹配。

因此本实现可以在 groups=1 / groups>1 / groups==in_channels (DW) 三种情形下
正确前向,适合用于 RGBlock 的 Conv 替换等场景。
"""

import math
import time

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_


# ----------------------------------------------------------------------------
# 基础激活模块
# ----------------------------------------------------------------------------
class StarReLU(nn.Module):
    """StarReLU 激活: s * relu(x) ** 2 + b。

    相比普通 ReLU 具有可学习的尺度与偏置,常用于注意力分支的非线性变换。
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        """初始化 StarReLU 的尺度/偏置参数。"""
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        """前向: 对输入应用 StarReLU 非线性。"""
        return self.scale * self.relu(x) ** 2 + self.bias


# ----------------------------------------------------------------------------
# KSM Global: 基于全局上下文生成 channel / filter / spatial / kernel 四类注意力
# ----------------------------------------------------------------------------
class KernelSpatialModulation_Global(nn.Module):
    """KSM 的全局分支,按 ODConv 范式输出多路注意力。

    注意:
        - in_planes 表示 "卷积核输入通道维" 的长度 (= in_channels // groups),
          用于 channel_fc 的输出通道数,使其与 kernel 权重的 cin 轴对齐。
        - in_planes_for_input 表示 "输入特征图的真实通道数" (= in_channels),
          用于 fc 的输入通道数。
        - groups: 当 in_planes == groups == out_planes 时视为 depth-wise conv,
          此时 filter 注意力退化为 skip。
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625,
                 kernel_num=4, min_channel=16, temp=1.0, kernel_temp=None,
                 kernel_att_init='dyconv_as_extra', att_multi=2.0, ksm_only_kernel_att=False,
                 att_grid=1, stride=1, spatial_freq_decompose=False, act_type='sigmoid',
                 in_planes_for_input=None):
        """构造 KSM_Global 所需要的各条注意力分支。"""
        super().__init__()
        if in_planes_for_input is None:
            in_planes_for_input = in_planes

        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.temperature = temp
        self.kernel_temp = kernel_temp

        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_grid = att_grid
        self.fc = nn.Conv2d(in_planes_for_input, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        self.spatial_freq_decompose = spatial_freq_decompose

        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(
                    attention_channel,
                    in_planes * 2 if self.kernel_size > 1 else in_planes,
                    1, bias=True,
                )
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        """对内部卷积/BN 做合适的初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)
        if hasattr(self, 'filter_fc') and isinstance(self.filter_fc, nn.Conv2d):
            nn.init.normal_(self.filter_fc.weight, std=1e-6)
        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            nn.init.normal_(self.channel_fc.weight, std=1e-6)

    def update_temperature(self, temperature):
        """动态更新注意力的 softmax/sigmoid 温度。"""
        self.temperature = temperature

    @staticmethod
    def skip(_):
        """占位函数,返回常量 1.0,用于禁用某一路注意力。"""
        return 1.0

    def get_channel_attention(self, x):
        """根据全局特征生成 channel 注意力,形状 (b, 1, 1, cin_g, 1, 1)。"""
        if self.act_type == 'sigmoid':
            channel_attention = torch.sigmoid(
                self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature
            ) * self.att_multi
        elif self.act_type == 'tanh':
            channel_attention = 1 + torch.tanh_(
                self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature
            )
        else:
            raise NotImplementedError
        return channel_attention

    def get_filter_attention(self, x):
        """根据全局特征生成 filter 注意力,形状 (b, 1, cout, 1, 1, 1)。"""
        if self.act_type == 'sigmoid':
            filter_attention = torch.sigmoid(
                self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature
            ) * self.att_multi
        elif self.act_type == 'tanh':
            filter_attention = 1 + torch.tanh_(
                self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature
            )
        else:
            raise NotImplementedError
        return filter_attention

    def get_spatial_attention(self, x):
        """根据全局特征生成 spatial 注意力,形状 (b, 1, 1, 1, k, k)。"""
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        """根据全局特征生成 kernel 基函数权重,形状 (b, kernel_num, 1, 1, 1, 1)。"""
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type == 'tanh':
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError
        return kernel_attention

    def forward(self, x, use_checkpoint=False):
        """前向: 可选地走 checkpoint 以节省显存。"""
        if use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        """实际前向: 先投影到注意力空间,再分别输出四路注意力。"""
        avg_x = self.relu(self.bn(self.fc(x)))
        return (
            self.func_channel(avg_x),
            self.func_filter(avg_x),
            self.func_spatial(avg_x),
            self.func_kernel(avg_x),
        )


# ----------------------------------------------------------------------------
# KSM Local: 借助 1D 卷积建模跨通道交互,生成逐通道逐空间的细粒度注意力
# ----------------------------------------------------------------------------
class KernelSpatialModulation_Local(nn.Module):
    """ECA 风格的局部 KSM 模块。

    Args:
        channel: 输入到本模块的通道数(对应 FDConv 的 in_channels_per_group)。
        kernel_num: 基函数数量。
        out_n: 每个 cin 位置要输出的展开维度(通常 = cout * k * k)。
        k_size: 1D 卷积核大小,会按 channel 自动调整。
        use_global: 是否额外启用 FFT 全局调制。
    """

    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        """构造 KSM_Local 内部的 Conv1d / LayerNorm 等。"""
        super().__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        if channel is not None:
            k_size = max(1, round((math.log2(max(channel, 2)) / 2) + 0.5) // 2 * 2 + 1)
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        self.use_global = use_global
        if self.use_global:
            self.complex_weight = nn.Parameter(
                torch.randn(1, self.channel // 2 + 1, 2, dtype=torch.float32) * 1e-6
            )
        self.norm = nn.LayerNorm(self.channel)

    def forward(self, x, x_std=None):
        """前向: 输入形状 (b, c, 1, 1),输出形状 (b, kn, c, out_n)。"""
        x = x.squeeze(-1).transpose(-1, -2)  # (b, 1, c)
        b, _, c = x.shape
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(
                torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1,
            )
        x = self.norm(x)
        att_logit = self.conv(x)
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c)  # (b, kn, out_n, c)
        att_logit = att_logit.permute(0, 1, 3, 2)  # (b, kn, c, out_n)
        return att_logit


# ----------------------------------------------------------------------------
# FBM: 频带调制模块
# ----------------------------------------------------------------------------
class FrequencyBandModulation(nn.Module):
    """FBM: 将输入特征分解为若干频带,并按频带动态加权。"""

    def __init__(self,
                 in_channels,
                 k_list=[2],
                 lowfreq_att=False,
                 fs_feat='feat',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 max_size=(64, 64),
                 **kwargs):
        """构造 FBM 的频带掩码与注意力卷积。"""
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.fs_feat = fs_feat
        self.act = act

        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group

        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:
                _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.spatial_group,
                    stride=1,
                    kernel_size=spatial_kernel,
                    groups=self.spatial_group,
                    padding=spatial_kernel // 2,
                    bias=True,
                )
                if init == 'zero':
                    nn.init.normal_(freq_weight_conv.weight, std=1e-6)
                    if freq_weight_conv.bias is not None:
                        freq_weight_conv.bias.data.zero_()
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError

        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        """一次性构造各频带的低通掩码,后续插值复用。"""
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        freq_indices = freq_indices.abs().max(dim=-1, keepdims=False)[0]

        masks = []
        for freq in k_list:
            mask = freq_indices < 0.5 / freq + 1e-8
            masks.append(mask)

        return torch.stack(masks, dim=0).unsqueeze(1)

    def sp_act(self, freq_weight):
        """将频带注意力 logit 通过指定激活函数映射到 (0, 2) 区间。"""
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            return 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError

    def forward(self, x, att_feat=None):
        """前向: 对不同频带分别做注意力调制后求和重建特征。"""
        if att_feat is None:
            att_feat = x

        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()
        b, _, h, w = x.shape

        x_fft = torch.fft.rfft2(x, norm='ortho')

        freq_h, freq_w = h, w // 2 + 1
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        for idx, freq in enumerate(self.k_list):
            mask = current_masks[idx]
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part

            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)

            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))

        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self.sp_act(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)

        return sum(x_list)


def get_fft2freq(d1, d2, use_rfft=False):
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)

    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    sorted_dist, indices = torch.sort(dist.view(-1))

    if use_rfft:
        d2 = d2 // 2 + 1
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    return sorted_coords.permute(1, 0), freq_hw


# ----------------------------------------------------------------------------
# FDConv: 频域动态卷积主模块
# ----------------------------------------------------------------------------
class FDConv(nn.Conv2d):
    """Frequency Dynamic Convolution(group-aware 版本)。

    - 继承 nn.Conv2d,因此构造签名与 API 完全兼容普通卷积。
    - 当 in_channels_per_group <= use_fdconv_if_c_gt 或 kernel_size 不在
      use_fdconv_if_k_in 中时,forward 会直接退化为 nn.Conv2d 标准前向。
    - 本版本修复了原实现在 groups > 1 时 kernel cin 轴 reshape 错误的 bug,
      所有涉及到"卷积核输入通道维"的位置均改用 in_channels // groups。
    """

    def __init__(self,
                 *args,
                 reduction=0.0625,
                 kernel_num=4,
                 use_fdconv_if_c_gt=16,
                 use_fdconv_if_k_in=[1, 3],
                 use_fdconv_if_stride_in=[1],
                 use_fbm_if_k_in=[3],
                 use_fbm_for_stride=False,
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 fbm_cfg={
                     'k_list': [2, 4, 8],
                     'lowfreq_att': False,
                     'fs_feat': 'feat',
                     'act': 'sigmoid',
                     'spatial': 'conv',
                     'spatial_group': 1,
                     'spatial_kernel': 3,
                     'init': 'zero',
                     'global_selection': False,
                 },
                 **kwargs):
        """初始化 FDConv 的超参、KSM/FBM 子模块以及 DFT 权重。"""
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.use_fdconv_if_stride_in = use_fdconv_if_stride_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in

        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']

        # group-aware: 卷积核真正的输入通道维
        self.in_channels_per_group = self.in_channels // self.groups
        # 防止 in_channels_per_group==1 导致 min(...)//2 == 0 的除 0 问题
        self._denom = max(1, min(self.out_channels, self.in_channels_per_group) // 2)

        # kernel_num & kernel_temp 的自适应
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp

        # 通道/核尺寸不满足条件时,直接走父类的普通 Conv2d
        # 使用 in_channels_per_group 而非总通道数,确保 DW 卷积时正确退化
        if self.in_channels_per_group <= self.use_fdconv_if_c_gt \
                or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return

        print('*** FDConv kernel_num:', self.kernel_num,
              '| in_channels_per_group:', self.in_channels_per_group,
              '| groups:', self.groups)

        # alpha 与 _denom 配合,保证与 dft_weight 归一化系数相互抵消
        self.alpha = self._denom * self.kernel_num * self.param_ratio / param_reduction

        # ---- KSM Global ----
        # in_planes(=in_channels_per_group)用于 channel_fc 的输出维度对齐 kernel cin 轴;
        # in_planes_for_input(=in_channels)用于 fc 输入的真实特征通道数。
        self.KSM_Global = KernelSpatialModulation_Global(
            in_planes=self.in_channels_per_group,
            out_planes=self.out_channels,
            kernel_size=self.kernel_size[0],
            groups=self.groups,
            temp=temp,
            kernel_temp=kernel_temp,
            reduction=reduction,
            kernel_num=self.kernel_num * self.param_ratio,
            kernel_att_init=None,
            att_multi=att_multi,
            ksm_only_kernel_att=ksm_only_kernel_att,
            act_type=self.ksm_global_act,
            att_grid=att_grid,
            stride=self.stride,
            spatial_freq_decompose=spatial_freq_decompose,
            in_planes_for_input=self.in_channels,
        )

        # ---- FBM ----
        # groups=1 且 spatial_group=1 时,所有通道共享同一频带调制(退化为标量缩放),
        # 自动将 spatial_group 设为 in_channels 以实现逐通道频带调制
        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            _fbm_cfg = dict(fbm_cfg)
            if self.groups == 1 and _fbm_cfg.get('spatial_group', 1) == 1:
                _fbm_cfg['spatial_group'] = self.in_channels
            self.FBM = FrequencyBandModulation(self.in_channels, **_fbm_cfg)

        # ---- KSM Local ----
        # group > 1 时 kernel cin=1(DW),此时 LayerNorm(1) 退化为常数输出,
        # 局部注意力失去意义,故直接关闭。
        if self.use_ksm_local and self.in_channels_per_group > 1:
            self.KSM_Local = KernelSpatialModulation_Local(
                channel=self.in_channels_per_group,
                kernel_num=1,
                out_n=int(self.out_channels * self.kernel_size[0] * self.kernel_size[1]),
            )
        else:
            self.use_ksm_local = False

        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param):
        """把 spatial-domain 权重转为 DFT 系数,并(可选)注册为可训练参数。"""
        d1 = self.out_channels
        d2 = self.in_channels_per_group
        k1, k2 = self.kernel_size[0], self.kernel_size[1]

        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))

        if self.param_reduction < 1:
            num_to_keep = max(1, int(freq_indices.size(1) * self.param_reduction))
            num_to_keep = max(self.kernel_num, (num_to_keep // self.kernel_num) * self.kernel_num)
            freq_indices = freq_indices[:, :num_to_keep]
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None, ].repeat(self.param_ratio, 1, 1) / self._denom
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(
                self.param_ratio, 1, 1, 1,
            ) / self._denom
            total_freqs = freq_indices.size(1)
            num_to_keep = (total_freqs // self.kernel_num) * self.kernel_num
            if num_to_keep < total_freqs:
                freq_indices = freq_indices[:, :num_to_keep]
                weight_rfft = weight_rfft[:, :num_to_keep, :]

        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight
        else:
            if self.linear_mode:
                assert self.kernel_size[0] == 1 and self.kernel_size[1] == 1
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)

        indices = []
        for i in range(self.param_ratio):
            indices.append(freq_indices.reshape(2, self.kernel_num, -1))
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)

    def get_FDW(self):
        """未转换到频域时,按当前 spatial weight 即时计算对应的 DFT 系数。"""
        d1 = self.out_channels
        d2 = self.in_channels_per_group
        k1, k2 = self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous()
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(
            self.param_ratio, 1, 1, 1,
        ) / self._denom
        return weight_rfft

    def forward(self, x):
        """前向: 根据通道/核尺寸条件,选择 FDConv 动态核卷积或退回普通卷积。"""
        if self.in_channels_per_group <= self.use_fdconv_if_c_gt \
                or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return super().forward(x)

        input_dtype = x.dtype

        cin_g = self.in_channels_per_group
        k1, k2 = self.kernel_size[0], self.kernel_size[1]

        # KSM 注意力计算: 保持与模型权重一致的 dtype,
        # AMP autocast 在训练时自动管理精度(将 sigmoid/tanh 等 upcast 到 float32),
        # 验证时模型和输入 dtype 一致不会冲突。
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)

        if self.use_ksm_local:
            if self.groups > 1:
                gx_for_local = global_x.view(x.size(0), self.groups, cin_g, 1, 1).mean(dim=1)
            else:
                gx_for_local = global_x

            hr_att_logit = self.KSM_Local(gx_for_local)
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, cin_g, self.out_channels, k1, k2)
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1

        batch_size, in_planes, height, width = x.size()
        b = batch_size

        # DFT_map: 按 cin_g 构建,与 FDW / indices 对齐;
        # view_as_complex 要求 float32/float64,始终使用 float32 以兼容 AMP。
        DFT_map = torch.zeros(
            (b, self.out_channels * k1, cin_g * k2 // 2 + 1, 2),
            device=x.device, dtype=torch.float32,
        )

        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)

        dft_weight = self.dft_weight if hasattr(self, 'dft_weight') else self.get_FDW()

        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1,
                )
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1,
                )

        # FFT 逆变换得到 float32 权重
        adaptive_weights = torch.fft.irfft2(
            torch.view_as_complex(DFT_map),
            dim=(1, 2),
            s=(self.out_channels * k1, cin_g * k2)
        ).reshape(batch_size, 1, self.out_channels, k1, cin_g, k2)
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)  # (b,1,cout,cin_g,k,k)

        # FBM 对输入做频带调制
        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
        aggregate_weight = torch.sum(aggregate_weight, dim=1)
        aggregate_weight = aggregate_weight.reshape(batch_size * self.out_channels, cin_g, k1, k2)

        # 转换为输入 dtype 后再卷积
        aggregate_weight = aggregate_weight.to(input_dtype)
        x_conv = x.reshape(1, -1, height, width).to(input_dtype)
        output = F.conv2d(
            x_conv, weight=aggregate_weight, bias=None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups * batch_size,
        )
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        if self.bias is not None:
            output = output + self.bias.to(output.dtype).view(1, -1, 1, 1)
        return output

    def profile_module(self, input: Tensor, *args, **kwargs):
        """保留自 mmdet 原实现的占位接口,供外部分析工具调用。"""
        b_sz, c, h, w = input.shape
        seq_len = h * w

        p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
        params = macs = getattr(self, 'hidden_size', c) * 2
        macs = macs * b_sz * seq_len

        return input, params, macs + m_ff


if __name__ == '__main__':
    for g in [1, 4, 64]:
        m = FDConv(in_channels=64, out_channels=64, kernel_size=3, padding=1,
                   groups=g, bias=True)
        y = m(torch.randn(2, 64, 16, 16))
        print(f'groups={g}, out shape: {y.shape}')