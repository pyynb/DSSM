"""
DSSM 双塔模型
"""

import sys

import torch.nn as nn

from utils.utils import get_activation

sys.path.append('../')
from model.EmbeddingModule import EmbeddingModule


class DSSM(nn.Module):
    def __init__(self, user_datatypes, item_datatypes, user_dnn_size=(256, 128),
                 item_dnn_size=(256, 128), dropout=0.0, activation='relu', use_senet=False):
        super().__init__()
        self.user_dnn_size = user_dnn_size
        self.item_dnn_size = item_dnn_size
        self.dropout = dropout
        self.user_datatypes = user_datatypes
        self.item_datatypes = item_datatypes

        self.embed_dim=128
        self.num_heads=4 
        self.num_layers=2
        self.dropout=0.1
        self.activation='relu'
        print("self.activation")
        print(self.activation)
        self.user_tower = TowerTransformer(user_datatypes, self.embed_dim, self.num_heads, self.num_layers, self.dropout, self.activation)
        self.item_tower = TowerTransformer(item_datatypes, self.embed_dim, self.num_heads, self.num_layers, dropout, activation)

        # 用户侧
        # self.user_tower = Tower(self.user_datatypes, self.user_dnn_size, self.dropout, activation=activation, use_senet=use_senet)
        # self.item_tower = Tower(self.item_datatypes, self.item_dnn_size, self.dropout, activation=activation, use_senet=use_senet)

    def forward(self, user_feat, item_feat):
        return self.user_tower(user_feat), self.item_tower(item_feat)


class Tower(nn.Module):
    def __init__(self, datatypes, dnn_size=(256, 128), dropout=0.0, activation='ReLU', use_senet=False):
        super().__init__()
        self.dnns = nn.ModuleList()
        self.embeddings = EmbeddingModule(datatypes, use_senet)
        input_dims = self.embeddings.sparse_dim + self.embeddings.dense_dim
        # self.use_senet = use_senet
        # if self.use_senet:
        #     self.se_net = SENet(input_dims)
        for dim in dnn_size:
            self.dnns.append(nn.Linear(input_dims, dim))
            # self.dnns.append(nn.BatchNorm1d(dim))
            self.dnns.append(nn.Dropout(dropout))
            self.dnns.append(get_activation(activation))
            input_dims = dim

    def forward(self, x):
        dnn_input = self.embeddings(x)
        # print(dnn_input.type())
        # if self.use_senet:
        #     dnn_input = self.se_net(dnn_input)

        # print(torch.mean(self.dnns[0].weight))
        for dnn in self.dnns:
            # if self.training == False:
            #     import pdb
            #     pdb.set_trace()
            dnn_input = dnn(dnn_input)

        # print('finish!')
        return dnn_input


import torch
import torch.nn as nn
from utils.utils import get_activation
from model.EmbeddingModule import EmbeddingModule

class TowerTransformer(nn.Module):
    def __init__(self, datatypes, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1, activation="ReLU"):
        """
        Transformer-based Tower
        :param datatypes: 输入特征类型，用于 EmbeddingModule
        :param embed_dim: Transformer 中的嵌入维度
        :param num_heads: 多头注意力机制的头数
        :param num_layers: Transformer Encoder 的层数
        :param dropout: Dropout 比例
        :param activation: 激活函数类型（如 'ReLU'）
        """
        super().__init__()
        self.embeddings = EmbeddingModule(datatypes,False)
        input_dim = self.embeddings.sparse_dim + self.embeddings.dense_dim

        # Embedding 映射到 Transformer 的输入维度
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation=activation.lower()
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最终输出映射
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass of the Transformer-based tower.
        :param x: 输入特征
        :return: 经过 Transformer 处理后的特征向量
        """
        # 1. Embedding 层
        embedded = self.embeddings(x)  # [batch_size, feature_dim]
        embedded = self.input_projection(embedded).unsqueeze(1)  # [batch_size, 1, embed_dim]

        # 2. Transformer Encoder
        transformer_output = self.transformer(embedded)  # [batch_size, seq_len=1, embed_dim]

        # 3. 提取输出特征（这里 seq_len=1，可以直接 squeeze）
        output = transformer_output.squeeze(1)  # [batch_size, embed_dim]

        # 4. 映射到最终输出
        return self.output_projection(output)
