import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utility.layers import GraphConvolution,Linear
from utility.utils import normalize_A, generate_cheby_adj
from utility.utils import ChannelwiseLayerNorm, ResBlock, Conv1D, ConvTrans1D
from utility import models


class Chebynet(nn.Module):
    def __init__(self, in_channel=128, k_adj=3):
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


class EEGEncoder(nn.Module):
    def __init__(self, num_electrodes=128, k_adj=3, enc_channel=128, feature_channel=64, kernel_size=8,
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
        self.layer1 = Chebynet(128, k_adj)
        self.projection = nn.Conv1d(128, feature_channel, self.proj_kernel_size, bias=False, stride=self.stride)
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
        L = normalize_A(self.A)
        output = self.layer1(spike, L)
        
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


class BASEN(nn.Module):
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
    def __init__(self, L1=0.0025, L2=0.01, L3=0.02, enc_channel=128, feature_channel=64, encoder_kernel_size=8, layers=4,
                rnn_type='LSTM', norm='ln', K=250, dropout=0, bidirectional=True, CMCA_kernel=3,
                CMCA_layer_num=3):
        super(BASEN, self).__init__()

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
        self.spike_encoder = EEGEncoder(enc_channel=enc_channel, feature_channel=feature_channel)
        # audio encoder
        #self.audio_encoder = AudioEncoder(self.encoder_kernel, self.enc_channel)
        self.encoder_1d_short = Conv1D(1, enc_channel, self.L1, stride=self.L1 // 2, padding=0)
        self.encoder_1d_middle = Conv1D(1, enc_channel, self.L2, stride=self.L1 // 2, padding=0)
        self.encoder_1d_long = Conv1D(1, enc_channel, self.L3, stride=self.L1 // 2, padding=0)
        self.ln = ChannelwiseLayerNorm(3*enc_channel)
        # n x N x T => n x O x T
        self.proj = Conv1D(3*enc_channel, enc_channel, 1)
        # DPRNN separation network
        self.DPRNN = models.DPRNN(self.enc_channel, self.enc_channel, self.feature_channel,
                              self.layer, self.CMCA_kernel ,rnn_type=rnn_type, norm=norm, K=self.K, dropout=dropout,
                               bidirectional=bidirectional,  CMCA_layer_num=CMCA_layer_num)
        
        # self.conv1d = nn.Conv1d(in_channels=64, out_channels=enc_channel,
        #                         kernel_size=self.encoder_kernel, stride=self.encoder_kernel//2, groups=1, bias=False)
        # output decoder
        self.decoder = ConvTrans1D(enc_channel, 1, self.L1 , stride=self.L1 // 2, bias=True)
        self.conv1d = nn.Conv1d(128,64,1)
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
        short = torch.sigmoid(self.DPRNN(w1, enc_output_spike)).view(batch_size, self.enc_channel, -1)
        middle = torch.sigmoid(self.DPRNN(w2, enc_output_spike)).view(batch_size, self.enc_channel, -1)
        long = torch.sigmoid(self.DPRNN(w3, enc_output_spike)).view(batch_size, self.enc_channel, -1)
        enc_output = self.ln(torch.cat([w1, w2, w3], 1))
        enc_output = self.proj(enc_output) #torch.Size([8, 128, 1624])
          # spike_input(8,128,29184) ---> enc_output_spike (8,32,29183)   kernel=8,(8,64,7295)
        #kernel_size=16时, enc_output_spike (8,64,3645)
        # generate masks
        masks = self.ln(torch.cat([short, middle, long], 1))
        masks = self.proj(masks)
        #print("shape:", torch.sigmoid(self.DPRNN(enc_output, enc_output_spike)).shape)
        #masks = torch.sigmoid(self.DPRNN(enc_output, enc_output_spike)).view(batch_size, self.num_spk, self.enc_channel, -1)
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
    nnet = BASEN()
    x = nnet(x, y)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
