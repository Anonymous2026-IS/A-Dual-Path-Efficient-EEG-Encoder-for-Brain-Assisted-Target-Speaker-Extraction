import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.layers import GraphConvolution,Linear
from utility.utils import normalize_A, generate_cheby_adj
from utility.utils import ChannelwiseLayerNorm, ResBlock, Conv1D, ConvTrans1D
from utility import models
class TEI_Module(nn.Module):
    """ Trainable EEG Interpolation (TEI) Module """
    def __init__(self, in_channels=32, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        # 分组转置卷积 (groups=in_channels实现通道独立处理)
        self.interpolate = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=scale_factor*2,  # 较大核尺寸以捕获更广的上下文
            stride=scale_factor,
            padding=scale_factor//2,
            groups=in_channels,  # 分组卷积减少参数量
            bias=False  # 无偏置项
        )
        # 初始化核权重为类线性插值
        self._init_weights()

    def _init_weights(self):
        # 初始化为类线性插值的权重
        with torch.no_grad():
            k = self.interpolate.kernel_size[0]
            weights = torch.linspace(0, 1, k).repeat(self.interpolate.in_channels, 1, 1)
            self.interpolate.weight.copy_(weights)

    def forward(self, x):
        # x: [batch, channels, time]
        return F.relu(self.interpolate(x))

class Chebynet(nn.Module):
    def __init__(self, in_channel=32, k_adj=3):
        super(Chebynet, self).__init__()
        self.K = k_adj
        self.gc = nn.ModuleList()
        for i in range(k_adj):
            self.gc.append(GraphConvolution(in_channel, in_channel))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        return F.relu(result)

class EEGEncoder_TEI(nn.Module):
    def __init__(self, num_electrodes=32, k_adj=3, enc_channel=32, feature_channel=64, 
                 kernel_size=8, scale_factor=4, rnn_type='LSTM', norm='ln', K=160, 
                 dropout=0, bidirectional=False, kernel=3, skip=True):
        super().__init__()
        
        # 超参数
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.scale_factor = scale_factor  # 上采样倍数 (EEG 128Hz -> 语音 16kHz需要约125倍，这里简化演示)
        
        # 可训练插值模块
        self.tei = TEI_Module(num_electrodes, scale_factor)
        
        # 图卷积部分 (保持原结构)
        self.BN1 = nn.BatchNorm1d(29184)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
        nn.init.xavier_normal_(self.A)
        self.layer1 = Chebynet(num_electrodes, k_adj)
        
        # 投影层 (调整通道数)
        self.projection = nn.Conv1d(
            num_electrodes, 
            feature_channel, 
            kernel_size=kernel_size, 
            stride=4,  # 下采样以平衡TEI的上采样
            bias=False
        )
        
        # 特征编码器 (保持原ResBlock结构)
        self.eeg_encoder = nn.Sequential(
            ChannelwiseLayerNorm(feature_channel),
            Conv1D(feature_channel, feature_channel, 1),
            ResBlock(feature_channel, feature_channel),
            ResBlock(feature_channel, enc_channel),
            ResBlock(enc_channel, enc_channel),
            Conv1D(enc_channel, feature_channel, 1),
        )

    def forward(self, x):
        # x: [batch, time, electrodes]
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # 保持您的原始归一化
        
        # 步骤1: 时间维度上采样 (TEI核心)
        # 转换维度为 [batch, electrodes, time]
        # x_tei = x.permute(0, 2, 1)      #x_tei shape: torch.Size([8, 29184, 32])
        # print("x_tei shape:", x_tei.shape)
        x_tei = x
        x_tei = self.tei(x_tei)  # 可学习插值
        print("x_tei shape:", x_tei.shape)  #x_tei shape: torch.Size([1, 32, 116736])
        
        # 步骤2: 图卷积处理
        L = normalize_A(self.A)
        x_gcn = self.layer1(x, L)  # 原始时序处理
        print("x_gcn shape1:", x_gcn.shape) #x_gcn shape1: torch.Size([1, 32, 29184])

        # 对齐维度 (可选: 将GCN输出也上采样，或TEI后下采样)
        # x_gcn = x_gcn.permute(0, 2, 1)
        # x_gcn = F.interpolate(x_gcn, size=x_tei.size(2), mode='linear')
        # print("x_gcn shape2:", x_gcn.shape)

        # 步骤3: 融合插值后特征和图特征
        x_fused = x_tei + x_gcn  # 简单相加融合
        
        # 继续原有流程
        x_proj = self.projection(x_fused)
        output = self.eeg_encoder(x_proj)
        
        return output  # [batch, channels, time]