import _init_paths

import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F
import attr
from mmcv.cnn import ConvModule
from lib.utils.ops import conv3x3_bn_relu, resize


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, embedding_dim, norm_layer, dropout, interpolate_mode, align_corners):
        super(SegFormerHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.dropout = nn.Dropout2d(dropout)
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.linear_project = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim // 4,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_saliency_pred = nn.Conv2d(embedding_dim // 4, 1, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        ab = self.linear_c4(c4)

        _c4 = self.linear_c4(c4).permute(0,2,1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = F.interpolate(_c, size=_c.shape[-1]*2, mode=self.interpolate_mode, align_corners=self.align_corners)
        _c = self.linear_project(_c)
        _c = F.interpolate(_c, size=_c.shape[-1]*2, mode=self.interpolate_mode, align_corners=self.align_corners)
        x = self.dropout(_c)
        x = self.linear_saliency_pred(x)

        return x
