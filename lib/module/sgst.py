import _init_paths

import torch
from torch import nn
from lib.module.blocks import *

class GenerateHeatMap(nn.Module):
    def __init__(self, in_channels=128):
        super(GenerateHeatMap, self).__init__()
        self.in_channels= in_channels
        self.sigmoid = nn.Sigmoid()
        self.sal_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x_embed):

        # Generate Saliency Map
        saliency_map = self.sal_conv(x_embed)
        heat_map = self.sigmoid(saliency_map)

        return heat_map, saliency_map

class SemanticGatheringScatteringTransformer(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            hidden_dim: int,
            nheads: int, # 8
            dim_feedforward: int, # 8*hidden_dim
            dec_layers: int,
            dropout: float,
            select_tokens: int,
            threshold: float
        ):

        super(SemanticGatheringScatteringTransformer, self).__init__()

        self.threshold = threshold
        self.generate_heat_map = GenerateHeatMap(in_channels)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.dropout = dropout
        self.fg_self_attention_layers = nn.ModuleList()
        self.bg_self_attention_layers = nn.ModuleList()
        self.fg_ffn_layers = nn.ModuleList()
        self.bg_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.fg_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.dropout,
                )
            )

            self.bg_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.dropout,
                )
            )

            self.fg_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.dropout,
                )
            )

            self.bg_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.dropout,
                )
            )

        self.fg_soft_weights = nn.Conv2d(in_channels, select_tokens, 1, 1)
        self.bg_soft_weights = nn.Conv2d(in_channels, select_tokens, 1, 1)
        self.project = nn.Conv2d(in_channels*2, in_channels, 1, 1)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.project(x)
        B, C, H, W = x.shape

        features = []
        for bs in range(0, B):
            feature = x[bs].unsqueeze(0)
            heat_map, saliency_map = self.generate_heat_map(feature)
            fg_index = torch.ge(heat_map, self.threshold)  # (bs, 1, H, W)
            bg_index = torch.lt(heat_map, 1 - self.threshold)  # (bs, 1, H, W)

            # all fg or all bg
            tensor0 = torch.tensor(0).to(feature.device).detach()
            if torch.sum(fg_index) == tensor0 or torch.sum(bg_index) == tensor0:
                features.append(feature)
            else:
                # fg or bg as quries
                fg_queries = torch.masked_select(feature, fg_index)
                bg_queries = torch.masked_select(feature, bg_index)
                fg_queries = fg_queries.view(C, -1).unsqueeze(0).permute(2, 0, 1)
                bg_queries = bg_queries.view(C, -1).unsqueeze(0).permute(2, 0, 1)
                fg_feature = feature * heat_map
                bg_feature = feature * (1 - heat_map)
                fg_soft_weights = self.fg_soft_weights(fg_feature).flatten(2)  # 1, K, H*W
                bg_soft_weights = self.bg_soft_weights(bg_feature).flatten(2)
                fg_feature = fg_feature.flatten(2)  # 1, C, H*W
                bg_feature = bg_feature.flatten(2)
                fg_feature = torch.einsum('bkn,bcn->bck', fg_soft_weights, fg_feature)
                bg_feature = torch.einsum('bkn,bcn->bck', bg_soft_weights, bg_feature)
                fg_feature = fg_feature.permute(2, 0, 1)
                bg_feature = bg_feature.permute(2, 0, 1)

                for i in range(self.num_layers):
                    fg_queries, fg_attn_weights = self.fg_self_attention_layers[i](
                        fg_queries, fg_feature, fg_feature, tgt_mask=None,
                        tgt_key_padding_mask=None, query_pos=None)

                    bg_queries, bg_attn_weights = self.bg_self_attention_layers[i](
                        bg_queries, bg_feature, bg_feature, tgt_mask=None,
                        tgt_key_padding_mask=None, query_pos=None)

                    fg_queries = self.fg_ffn_layers[i](
                        fg_queries, (fg_queries.shape[0], 1))

                    bg_queries = self.bg_ffn_layers[i](
                        bg_queries, (bg_queries.shape[0], 1))

                feature.clone().masked_scatter_(fg_index, fg_queries)
                feature.clone().masked_scatter_(bg_index, bg_queries)

                features.append(feature)

        z = torch.stack(features, 0).squeeze(1)

        return z

