import torch
import torch.nn as nn
import torch.nn.functional as F

def global_median_pooling(x):
    """全局中值池化操作"""
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, bias=True)

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)

        avg_out = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_pool), inplace=True)))
        max_out = torch.sigmoid(self.fc2(F.relu(self.fc1(max_pool), inplace=True)))
        median_out = torch.sigmoid(self.fc2(F.relu(self.fc1(median_pool), inplace=True)))

        return avg_out + max_out + median_out

class GMEA(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4, groups=4):
        super(GMEA, self).__init__()
        assert in_channels == out_channels
        self.C = in_channels
        self.O = out_channels

        self.groups = groups

        self.channel_attention = ChannelAttention(input_channels=in_channels,
                                                  internal_neurons=in_channels // channel_attention_reduce)

        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

        self.depth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])

        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()


    def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batchsize, -1, height, width)

    def forward(self, inputs):
        inputs = self.pointwise_conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.channel_attention(inputs)
        inputs = channel_att_vec * inputs

        inputs = self.channel_shuffle(inputs, self.groups)

        initial_out = self.initial_depth_conv(inputs)

        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        spatial_att = self.pointwise_conv(spatial_out)
        out = spatial_att * inputs
        out = self.pointwise_conv(out)
        return out


