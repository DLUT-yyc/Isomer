import _init_paths

import lib
import torch
import torch.nn as nn

class VOSNet(nn.Module):
    def __init__(self, opt):
        super(VOSNet, self).__init__()
        self.opt = opt
        self.bn = nn.BatchNorm2d
        self.num_points = opt.num_points

        if opt.encoder == 'swin_tiny': 
            self.backbone_x = lib.swin_tiny()
            self.backbone_y = lib.swin_tiny()
        elif opt.encoder == 'mit_b0':
            self.backbone_x = lib.mit_b0()
            self.backbone_y = lib.mit_b0()

        if opt.encoder == 'swin_tiny':
            feature_channels = [96, 192, 384, 768]
            embedding_dim = 192
        if opt.encoder == 'mit_b0':
            feature_channels = [32, 64, 160, 256]
            embedding_dim = 64

        self.decode_head = lib.SegFormerHead(feature_channels, embedding_dim, self.bn, opt.seghead_dropout, 'bilinear', False)

        self.fusion_module0 = lib.ContextSharingTransformer(feature_channels[0], feature_channels[0]*opt.ffn_dim_ratio, opt.fusion_module_dropout)
        self.fusion_module1 = lib.ContextSharingTransformer(feature_channels[1], feature_channels[1]*opt.ffn_dim_ratio, opt.fusion_module_dropout)
        self.fusion_module2 = lib.SemanticGatheringScatteringTransformer(feature_channels[2], feature_channels[2], opt.num_attn_heads, feature_channels[2]*opt.ffn_dim_ratio, opt.num_blocks, opt.fusion_module_dropout, opt.num_points, opt.threshold)
        self.fusion_module3 = lib.SemanticGatheringScatteringTransformer(feature_channels[3], feature_channels[3], opt.num_attn_heads, feature_channels[3]*opt.ffn_dim_ratio, opt.num_blocks, opt.fusion_module_dropout, opt.num_points // 4, opt.threshold)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.init_backbone()

    def init_backbone(self):
        if self.opt.encoder == 'swin_tiny':
            saved_state_dict = torch.load('./pretrained_model/swin_tiny_patch4_window7_224.pth', map_location='cpu')
        if self.opt.encoder == 'mit_b0':
            saved_state_dict = torch.load('./pretrained_model/mit_b0.pth', map_location='cpu')
        
        if 'swin' in self.opt.encoder:
            self.backbone_x.load_state_dict(saved_state_dict['model'], strict=False)
            self.backbone_y.load_state_dict(saved_state_dict['model'], strict=False)
        elif 'mit' in self.opt.encoder:
            self.backbone_x.load_state_dict(saved_state_dict, strict=False)
            self.backbone_y.load_state_dict(saved_state_dict, strict=False)

    def forward(self, x, y):

        x_0, x_1, x_2, x_3 = self.backbone_x(x)
        y_0, y_1, y_2, y_3 = self.backbone_y(y)

        z_0 = self.fusion_module0(x_0, y_0)
        z_1 = self.fusion_module1(x_1, y_1)
        z_2 = self.fusion_module2(x_2, y_2)
        z_3 = self.fusion_module3(x_3, y_3)

        z = self.decode_head([z_0, z_1, z_2, z_3])

        return z

