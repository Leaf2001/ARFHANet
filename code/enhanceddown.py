import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class GaussianPyramid(nn.Module):
    def __init__(self, in_channels, num_levels=4):
        super(GaussianPyramid, self).__init__()
        self.num_levels = num_levels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        pyr = [x]
        current = x
        for _ in range(self.num_levels - 1):
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            pyr.append(current)
        return pyr

class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels, num_levels=4):
        super(LaplacianPyramid, self).__init__()
        self.num_levels = num_levels
        self.gaussian_pyramid = GaussianPyramid(in_channels, num_levels=num_levels)

    def forward(self, x):
        gaussian_pyr = self.gaussian_pyramid(x)
        laplacian_pyr = []
        for i in range(self.num_levels - 1):
            upsampled = F.interpolate(gaussian_pyr[i + 1], scale_factor=2, mode='bilinear', align_corners=False)
            laplacian = gaussian_pyr[i] - upsampled
            laplacian_pyr.append(laplacian)
        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr



class EnhancedDownsampling(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, pyramid_levels=4):
        super(EnhancedDownsampling, self).__init__()
        enhancement_channels = in_channel//16
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.lap_pyramid = LaplacianPyramid(in_channel, num_levels=pyramid_levels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enhance_HL = nn.Conv2d(in_channel, enhancement_channels, kernel_size=3, padding=1)
        self.enhance_LH = nn.Conv2d(in_channel, enhancement_channels, kernel_size=3, padding=1)
        self.enhance_HH = nn.Conv2d(in_channel, enhancement_channels, kernel_size=3, padding=1)

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 小波变换
        yL, yH = self.wt(x)

        y_HL = yH[0][:, :, 0, :]  # 水平高频
        y_LH = yH[0][:, :, 1, :]  # 垂直高频
        y_HH = yH[0][:, :, 2, :]  # 对角线高频

        pyr_HL = self.lap_pyramid(y_HL)
        pyr_LH = self.lap_pyramid(y_LH)
        pyr_HH = self.lap_pyramid(y_HH)

        pyr_HL = [F.interpolate(level, size=pyr_HL[0].size()[2:], mode='bilinear', align_corners=False) for level in
                  pyr_HL]
        pyr_LH = [F.interpolate(level, size=pyr_LH[0].size()[2:], mode='bilinear', align_corners=False) for level in
                  pyr_LH]
        pyr_HH = [F.interpolate(level, size=pyr_HH[0].size()[2:], mode='bilinear', align_corners=False) for level in
                  pyr_HH]


        enhanced_HL = [self.enhance_HL(level) for level in pyr_HL]
        enhanced_LH = [self.enhance_LH(level) for level in pyr_LH]
        enhanced_HH = [self.enhance_HH(level) for level in pyr_HH]

        enhanced_HL = torch.cat(enhanced_HL, dim=1)
        enhanced_LH = torch.cat(enhanced_LH, dim=1)
        enhanced_HH = torch.cat(enhanced_HH, dim=1)

        reduce_channels = nn.Conv2d(yL.size(1), yL.size(1) // 4, kernel_size=1)
        yL = reduce_channels(yL)
        enhanced_features = torch.cat([enhanced_HL, enhanced_LH, enhanced_HH,yL], dim=1)

        output = self.conv_bn_relu(enhanced_features)
        return output

