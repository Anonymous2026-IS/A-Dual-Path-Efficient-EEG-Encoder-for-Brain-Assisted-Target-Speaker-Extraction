import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utility.layers import GraphConvolution,Linear
from utility.utils import normalize_A, generate_cheby_adj
from utility.utils import ChannelwiseLayerNorm, ResBlock, Conv1D, ConvTrans1D
from utility import models
from utility.layers import GraphConvolution,Linear
from utility.utils import normalize_A, generate_cheby_adj
from utility.utils import ChannelwiseLayerNorm, ResBlock, Conv1D, ConvTrans1D
from utility import models
from EEGEncoder_RWKV import Block
class EEGEncoder_rwkv_v1(nn.Module):
    def __init__(self, num_electrodes=32, k_adj=3, enc_channel=32, feature_channel=64, kernel_size=8,
                 rnn_type='LSTM', norm='ln', K=160, dropout=0, bidirectional=False, kernel=3, skip=True):
        super(EEGEncoder_rwkv_v1, self).__init__()
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.kernel = kernel
        self.proj_kernel_size = kernel_size
        self.stride = 4
        self.K = K

        self.BN1 = nn.BatchNorm1d(29184)
        self.layer1 = Chebynet(32, k_adj)
        self.projection = nn.Conv1d(32, feature_channel, self.proj_kernel_size, bias=False, stride=self.stride)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.xavier_normal_(self.A)

        self.eeg_encoder = nn.Sequential(
            ChannelwiseLayerNorm(feature_channel),
            Conv1D(feature_channel, feature_channel, 1),
            ResBlock(feature_channel, feature_channel),
            ResBlock(feature_channel, enc_channel),
            ResBlock(enc_channel, enc_channel),
            Conv1D(enc_channel, feature_channel, 1),
        )

        # 改进的 配置
        self.dp_eeg_rwkv = nn.Sequential(
            # 维度转换适配层
            nn.Linear(feature_channel, feature_channel * 2),
            nn.GELU(),
            
            # Block
            Block(
                n_embd=feature_channel * 2,  # 扩大通道数
                n_layer=2,                   # 单层简化
                layer_id=0,
                shift_pixel=0,
                shift_mode='q_shift',
                channel_gamma=1/2,           # 调整gamma值
                drop_path=0.1,               # 添加dropout
                hidden_rate=2,               # 降低隐藏层扩展率
                init_mode='fancy',
                post_norm=True,              # 启用后归一化
                key_norm=True,               # 启用key归一化
            ),
            
            # 输出适配层
            nn.Linear(feature_channel * 2, feature_channel),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, spike):
        spike = self.BN1(spike.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        output = self.layer1(spike, L)
        output = self.projection(output)
        output = self.eeg_encoder(output)  # [B, C, T]

        # 转换为 Block 需要的格式 [B, T, C]
        output = output.transpose(1, 2)

        # 保存残差连接
        identity = output
        
        # 通过 模块
        output = self.dp_eeg_rwkv(output)
        
        # 残差连接
        output = output + identity
        
        # 转回原始维度 [B, C, T]
        output = output.transpose(1, 2)
        
        return output
    
class EEGEncoder_rwkv(nn.Module):
    def __init__(self, num_electrodes=128, k_adj=3, enc_channel=128, feature_channel=64, kernel_size=8,
                 rnn_type='LSTM', norm='ln', K=160, dropout=0, bidirectional=False, kernel=3, skip=True):
        super(EEGEncoder_rwkv, self).__init__()
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.kernel = kernel
        self.proj_kernel_size = kernel_size
        self.stride = 4
        self.K = K

        self.BN1 = nn.BatchNorm1d(29184)
        self.layer1 = Chebynet(128, k_adj)
        self.projection = nn.Conv1d(128, feature_channel, self.proj_kernel_size, bias=False, stride=self.stride)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.xavier_normal_(self.A)

        self.eeg_encoder = nn.Sequential(
            ChannelwiseLayerNorm(feature_channel),
            Conv1D(feature_channel, feature_channel, 1),
            ResBlock(feature_channel, feature_channel),
            ResBlock(feature_channel, enc_channel),
            ResBlock(enc_channel, enc_channel),
            Conv1D(enc_channel, feature_channel, 1),
        )

        # 添加 Block（EEG 版）
        self.dp_eeg_rwkv = Block(
            n_embd=feature_channel,  # 必须和输入通道数一致
            n_layer=2,               # 这里随便写，只有一个 Block
            layer_id=0,
            shift_pixel=0,           # EEG 用不到图像 shift
            shift_mode='q_shift',    # 不启用 shift_pixel 时无影响
            channel_gamma=1/4,
            drop_path=0.,
            hidden_rate=4,
            init_mode='fancy',
            post_norm=False,
            key_norm=False,
            init_values=None,
            with_cp=False
        )

    def forward(self, spike):
        spike = self.BN1(spike.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        output = self.layer1(spike, L)
        output = self.projection(output)
        output = self.eeg_encoder(output)  # [B, C, T]

        # 转换为 Block 需要的格式 [B, T, C]
        output = output.transpose(1, 2)

        # 这里 patch_resolution 设为 (1, T) 占位
        output = self.dp_eeg_rwkv(output, patch_resolution=(1, output.shape[1]))

        # 如果后面还要用 1D conv，可以转回 [B, C, T]
        output = output.transpose(1, 2)

        return output

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

    def forward(self, x ,L):
        
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result

class EEGEncoder_TEI(nn.Module):
    def __init__(self, num_electrodes=32, k_adj=3, enc_channel=32, feature_channel=64, 
                 kernel_size=8, scale_factor=1, rnn_type='LSTM', norm='ln', K=160, 
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
        # print("x_tei shape:", x_tei.shape)  #x_tei shape: torch.Size([1, 32, 116736])
        x_tei = x_tei[:, :, :-1]

        # 步骤2: 图卷积处理
        L = normalize_A(self.A)
        x_gcn = self.layer1(x, L)  # 原始时序处理
        # print("x_gcn shape1:", x_gcn.shape) #x_gcn shape1: torch.Size([1, 32, 29184])

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
    
class EEGEncoder(nn.Module):
    def __init__(self, num_electrodes=32, k_adj=3, enc_channel=32, feature_channel=64, kernel_size=8,
                 rnn_type='LSTM', norm='ln', K=160,  dropout=0, bidirectional=False, kernel=3, skip=True):
        super(EEGEncoder, self).__init__()
        # hyper parameters
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.kernel = kernel
        self.proj_kernel_size = kernel_size
        self.stride = 4
        self.K=k_adj
        #self.layer = layer
        self.K = K
        self.BN1 = nn.BatchNorm1d(29184)
        self.layer1 = Chebynet(32, k_adj)
        self.projection = nn.Conv1d(32, feature_channel, self.proj_kernel_size, bias=False, stride=self.stride)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes , num_electrodes).cuda())
        nn.init.xavier_normal_(self.A)
        # self.conv1d = nn.Conv1d(feature_channel, feature_channel, 1, bias=False)
        # self.linear =nn.Linear(enc_channel, enc_channel, bias=False)   
        # self.conv2d = nn.Conv2d(feature_channel, feature_channel, kernel_size=1)

        self.eeg_encoder = nn.Sequential(
            ChannelwiseLayerNorm(feature_channel),
            Conv1D(feature_channel, feature_channel, 1),
            ResBlock(feature_channel, feature_channel),
            ResBlock(feature_channel, enc_channel),
            ResBlock(enc_channel, enc_channel),
            Conv1D(enc_channel, feature_channel, 1),
        )
        # self.output = nn.Sequential(nn.PReLU(),
        #                             nn.Conv1d(feature_channel, feature_channel, 1)
        #                             )

    def forward(self, spike):


        spike = self.BN1(spike.transpose(1, 2)).transpose(1, 2)
        # print("spike.shape:",spike.shape)   #spike.shape: torch.Size([1, 32, 29184])
        L = normalize_A(self.A)
        output = self.layer1(spike, L)
        # print("output1:", output.shape) #output1: torch.Size([1, 32, 29184])

        output = self.projection(output)
        output = self.eeg_encoder(output)



        return output

class AudioEncoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters


        Example
        -------
        x = torch.randn(2, 1000)
        encoder = Encoder(kernel_size=4, out_channels=64)
        h = encoder(x)
        h.shape
        torch.Size([2, 64, 499])


    '''
    # kernel_size=16, out_channels=256, in_channels=1
    def __init__(self, kernel_size=8, enc_channels=64):
        super(AudioEncoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=enc_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1, bias=False)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x
    
class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):  # x (4,128,3652)
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class DP_EEG_TSE(nn.Module):
    """
        The Brain-Assisted Speech Enhancement Network, This is adapted from Conv-TasNet.

        args:
        enc_channel: The encoder channel num
        feature_channel: The hidden channel num in separation network
        encoder_kernel_size: Kernel size of the audio encoder and eeg encoder
        layer_per_stack: layer num of every stack
        stack: The num of stacks
        kernel: Kernel size in separation network
    """
    def __init__(self, L1=0.0025, L2=0.01, L3=0.02, enc_channel=32, feature_channel=64, encoder_kernel_size=8, layers=4,
                rnn_type='LSTM', norm='ln', K=250, dropout=0, bidirectional=True, CMCA_kernel=3,
                CMCA_layer_num=3):
        super(DP_EEG_TSE, self).__init__()

        # hyper parameters
        #self.num_spk = num_spk
        self.L1 = int(L1 * 14700)
        self.L2 = int(L2 * 14700)
        self.L3 = int(L3 * 14700)
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.num_spk = 1
        self.encoder_kernel = encoder_kernel_size
        self.CMCA_kernel = CMCA_kernel
        self.stride = 16
        self.win = 64
        self.layer = layers
        self.K = K
        #self.kernel = kernel
        # EEG encoder
        # self.spike_encoder = EEGEncoder(enc_channel=enc_channel, feature_channel=feature_channel)
        # self.spike_encoder = EEGEncoder_TEI(enc_channel=enc_channel, feature_channel=feature_channel)
        self.spike_encoder = EEGEncoder_rwkv(enc_channel=enc_channel, feature_channel=feature_channel)
        # self.spike_encoder = EEGEncoder_rwkv_v1(enc_channel=enc_channel, feature_channel=feature_channel)
        # audio encoder
        #self.audio_encoder = AudioEncoder(self.encoder_kernel, self.enc_channel)
        self.encoder_1d_short = Conv1D(1, enc_channel, self.L1, stride=self.L1 // 2, padding=0)
        self.encoder_1d_middle = Conv1D(1, enc_channel, self.L2, stride=self.L1 // 2, padding=0)
        self.encoder_1d_long = Conv1D(1, enc_channel, self.L3, stride=self.L1 // 2, padding=0)
        self.ln = ChannelwiseLayerNorm(3*enc_channel)
        # n x N x T => n x O x T
        self.proj = Conv1D(3*enc_channel, enc_channel, 1)
        # separation network
        self.separation = models.separation(self.enc_channel, self.enc_channel, self.feature_channel,
                              self.layer, self.CMCA_kernel ,rnn_type=rnn_type, norm=norm, K=self.K, dropout=dropout,
                               bidirectional=bidirectional,  CMCA_layer_num=CMCA_layer_num)

        # self.conv1d = nn.Conv1d(in_channels=64, out_channels=enc_channel,
        #                         kernel_size=self.encoder_kernel, stride=self.encoder_kernel//2, groups=1, bias=False)
        # output decoder
        self.decoder = ConvTrans1D(enc_channel, 1, self.L1 , stride=self.L1 // 2, bias=True)
        self.conv1d = nn.Conv1d(32,64,1)
    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input, spike_input):  #input(4,1,29184)  spike_input(4,128,29184)
        
        # padding
        output, rest = self.pad_signal(input)  #input(8,1,29184) --->output(8,1,29187)    (8,1,29264) rest48
        #spike, rest = self.pad_signal(spike_input)
        batch_size = output.size(0)
        enc_output_spike = self.spike_encoder(spike_input) 
        # waveform encoder
        #enc_output = self.audio_encoder(output)  # B, N, L  # output (8,1,29187) ---> enc_output (8,64,29186)   kernel_size=16时, enc_output (8,128,3647)   kernel=8,(8,128,7315)
        w1 = F.relu(self.encoder_1d_short(output))
        T = w1.shape[-1]
        xlen1 = output.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(self.encoder_1d_middle(F.pad(output, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_1d_long(F.pad(output, (0, xlen3 - xlen1), "constant", 0)))
        short = torch.sigmoid(self.separation(w1, enc_output_spike)).view(batch_size, self.enc_channel, -1)
        middle = torch.sigmoid(self.separation(w2, enc_output_spike)).view(batch_size, self.enc_channel, -1)
        long = torch.sigmoid(self.separation(w3, enc_output_spike)).view(batch_size, self.enc_channel, -1)
        enc_output = self.ln(torch.cat([w1, w2, w3], 1))
        enc_output = self.proj(enc_output) #torch.Size([8, 128, 1624])
          # spike_input(8,128,29184) ---> enc_output_spike (8,32,29183)   kernel=8,(8,64,7295)
        #kernel_size=16时, enc_output_spike (8,64,3645)
        # generate masks
        masks = self.ln(torch.cat([short, middle, long], 1))
        masks = self.proj(masks)
        #print("shape:", torch.sigmoid(self.separation(enc_output, enc_output_spike)).shape)
        #masks = torch.sigmoid(self.separation(enc_output, enc_output_spike)).view(batch_size, self.num_spk, self.enc_channel, -1)
        # masks = self.conv1d(masks).view(batch_size, self.num_spk, self.enc_channel, -1)  # B, C, N, L
        # masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L # 8,1,128,1624
        masked_output = enc_output * masks
        # waveform decoder
        # output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_channel, -1))
        output = self.decoder(masked_output)  # B*C, 1, L
        output = F.pad(output, (0, xlen1 - output.size(1)), "constant", 0)
        output = torch.unsqueeze(output, dim=1)   #29250
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        pad_spike = F.pad(enc_output_spike, (0, 1624-270)).view(batch_size, -1)
        audio_feature = self.conv1d(enc_output).reshape(batch_size, -1)
        return output, pad_spike, audio_feature

def test_conv_tasnet():
    x = torch.rand(8, 1, 29180)
    y = torch.rand(8, 128, 29180)
    nnet = DP_EEG_TSE()
    x = nnet(x, y)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
