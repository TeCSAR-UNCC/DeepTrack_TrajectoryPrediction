import torch.nn.functional as F
from torch import nn
from .tcn_2d import TemporalConvNet
from .config import config


class TCN(nn.Module):
    def __init__(self, input_size, output_size, kernel_sizes, dilation_sizes, input_scaling, num_channels, dropout=0.3, first_layer_chn=1, conv2d=False):
        super(TCN, self).__init__()
        self.output_size = output_size
        self.conv2d = conv2d
        self.tcn = TemporalConvNet(input_size, kernel_sizes, dilation_sizes,
                                   input_scaling, num_channels, dropout=dropout, conv2d=conv2d, first_layer_chn=first_layer_chn)
        #if conv2d:
        #    self.downsampler = nn.AdaptiveMaxPool2d((1, 1))
        #else:
        #    self.downsampler = nn.AdaptiveMaxPool1d(1)

        #self.prediction = nn.Linear(
        #    num_channels[-1], output_size, bias=False)
        #self.activation = nn.LeakyReLU()

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.tcn(inputs)
        #x = self.downsampler(x)
        #x = x.reshape(x.shape[0], -1)
        #x = flatten(x, 1)

        #x = self.prediction(x)
        #x = self.activation(x)

        return x


def get_network(net_configuration, input_length, prediction_size, dropout, use_conv2d, first_layer_chn=1):
    mc = net_configuration
    return TCN(input_size=input_length, output_size=prediction_size, kernel_sizes=mc['skrn'],
               dilation_sizes=mc['sdil'], input_scaling=mc['input_scaling'], num_channels=mc['nhid'],
               dropout=dropout, conv2d=use_conv2d, first_layer_chn=first_layer_chn)
