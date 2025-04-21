import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


def frequency_highpass_filter(x):
    B, C, H, W = x.shape
    freq = torch.fft.fft2(x, norm='ortho')
    freq_shift = torch.fft.fftshift(freq)

    u = torch.arange(H).to(x.device) - H // 2
    v = torch.arange(W).to(x.device) - W // 2
    U, V = torch.meshgrid(u, v, indexing='ij')
    D = torch.sqrt(U ** 2 + V ** 2)

    D0 = max(H, W) // 20
    highpass_filter = 1 - torch.exp(- (D ** 2) / (2 * D0 ** 2))
    highpass_filter = highpass_filter.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

    filtered_freq = freq_shift * highpass_filter
    filtered_freq = torch.fft.ifftshift(filtered_freq)
    x_filtered = torch.fft.ifft2(filtered_freq, norm='ortho').real
    return x_filtered


def frequency_lowpass_filter(x):
    B, C, H, W = x.shape
    freq = torch.fft.fft2(x, norm='ortho')
    freq_shift = torch.fft.fftshift(freq)

    u = torch.arange(H).to(x.device) - H // 2
    v = torch.arange(W).to(x.device) - W // 2
    U, V = torch.meshgrid(u, v, indexing='ij')
    D = torch.sqrt(U ** 2 + V ** 2)

    D0 = max(H, W) // 20

    lowpass_filter = torch.exp(- (D ** 2) / (2 * D0 ** 2))
    lowpass_filter = lowpass_filter.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

    filtered_freq = freq_shift * lowpass_filter
    filtered_freq = torch.fft.ifftshift(filtered_freq)
    x_filtered = torch.fft.ifft2(filtered_freq, norm='ortho').real
    return x_filtered


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        se_weight = self.global_avg_pool(x).view(b, c)
        se_weight = self.fc(se_weight).view(b, c, 1, 1)
        return x * se_weight


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))  # 计算每个像素的重要性
        return x * attention


class WFED(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(WFED, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt = DWTInverse(wave='haar', mode='zero')

        # 通道降维
        self.reduce_channels_HL = nn.Conv2d(in_channel, in_channel // 4, kernel_size=1)
        self.reduce_channels_LH = nn.Conv2d(in_channel, in_channel // 4, kernel_size=1)
        self.reduce_channels_HH = nn.Conv2d(in_channel, in_channel // 4, kernel_size=1)

        self.se_HL = SEBlock(in_channel // 4)
        self.pa_HL = PixelAttention(in_channel // 4)
        self.se_LH = SEBlock(in_channel // 4)
        self.pa_LH = PixelAttention(in_channel // 4)
        self.se_HH = SEBlock(in_channel // 4)
        self.pa_HH = PixelAttention(in_channel // 4)

        self.se_L = SEBlock(out_channel)
        self.sigmoid_L = nn.Sigmoid()

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL, y_LH, y_HH = yH[0][:, :, 0, :], yH[0][:, :, 1, :], yH[0][:, :, 2, :]

        enhanced_HL = self.pa_HL(self.se_HL(frequency_highpass_filter(y_HL)))
        enhanced_LH = self.pa_LH(self.se_LH(frequency_highpass_filter(y_LH)))
        enhanced_HH = self.pa_HH(self.se_HH(frequency_highpass_filter(y_HH)))

        yL = self.se_L(frequency_lowpass_filter(yL))
        yL = yL * self.sigmoid_L(yL)

        enhanced_HL = self.reduce_channels_HL(enhanced_HL)
        enhanced_LH = self.reduce_channels_LH(enhanced_LH)
        enhanced_HH = self.reduce_channels_HH(enhanced_HH)

        yH_new = torch.stack([enhanced_HL, enhanced_LH, enhanced_HH], dim=2)
        enhanced_features = self.conv_bn_relu(yL)

        output = self.iwt((enhanced_features, [yH_new]))

        residual = F.avg_pool2d(x, kernel_size=2, stride=2)
        output += residual

        return output
