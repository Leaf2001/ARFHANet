import torch
import torch.nn as nn
# 提出的密集嵌套注意力网络( DNA-Net )。
# ( a )特征提取模块。输入图像首先被送入密集嵌套交互模块( DNIM )，以聚合来自多个尺度的信息。
    # 值得注意的是，不同语义层次的特征通过通道和空间注意力模块( channel and spatial attention module，CSAM )进行自适应增强。
# ( b )特征金字塔融合模块( FPFM )。对增强后的特征进行上采样和拼接，实现多层输出融合。
# ( c )八连通邻域聚类算法。对分割图进行聚类，确定每个目标区域的质心。

# VGG_CBAM_Block：这是一个基于 VGG 的卷积块，结合了 CBAM（通道和空间注意力模块）。
# 该模块由两个卷积层、两个批归一化层、两个 ReLU 激活函数以及一个通道注意力（Channel Attention）和一个空间注意力（Spatial Attention）模块组成。
class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels) # 通道注意力模块，用于加权特征图的通道。
        self.sa = SpatialAttention() # 空间注意力模块，用于加权特征图的空间位置。

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out # 计算通道注意力系数，并通过逐元素相乘调整特征图的通道权重。
        out = self.sa(out) * out # 计算空间注意力系数，并通过逐元素相乘调整特征图的空间权重。
        out = self.relu(out)
        return out


# ChannelAttention：通道注意力模块，通过全局平均池化和全局最大池化来生成通道注意力图，帮助网络更好地关注重要的通道
# 在特征图中，不同的通道可能包含不同的有用信息，而通过通道注意力模块，网络能够自动调整每个通道的权重，从而增强对有用特征的关注。
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 这两个池化层分别执行全局平均池化和全局最大池化，将每个通道的特征图缩减为单个值，生成一个通道描述向量。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 这两个卷积层用于生成通道注意力权重。fc1 将输入通道数减少到 in_planes // 16，fc2 再将其恢复到 in_planes。
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 全局平均池化后，经过卷积层 fc1 和激活层 ReLU，再经过卷积层 fc2
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大平均池化后，经过卷积层 fc1 和激活层 ReLU，再经过卷积层 fc2
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# SpatialAttention：空间注意力模块，通过对输入特征图的平均池化和最大池化，生成空间注意力图，帮助网络关注重要的空间位置。
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 根据 kernel_size 确定填充的大小。卷积核大小为 7 时填充为 3，大小为 3 时填充为 1。这确保卷积操作后输出的空间尺寸保持一致。
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 通过在 dim=1（即通道维度）上对输入特征图 x 进行平均池化得到，这有助于提取全局平均的空间信息。
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 通过在 dim=1 上对输入特征图 x 进行最大池化得到，这有助于提取最显著的空间特征。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均池化和最大池化的结果在 dim=1 上进行拼接，形成一个新的特征图 x。
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# Res_CBAM_block：结合了残差连接的 CBAM 模块，包含两个卷积层、批归一化层、ReLU 激活函数，以及通道和空间注意力机制。
# 该模块可以处理输入特征图的通道数变化和步幅变化，并通过残差连接提高网络的表达能力。
class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 创建残差连接
        if stride != 1 or out_channels != in_channels:
            # 使用 nn.Sequential 创建一个顺序容器
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        # residual 用于存储残差连接的输出，如果不需要残差连接，则 residual 就是输入 x。
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual # 通过 out += residual 将残差连接的输出添加到主路径的输出上。
        out = self.relu(out)
        return out

# DNANet：整个 DNA-Net 网络结构，包括特征提取、特征金字塔融合以及深度监督。
# 该网络包括多个卷积层，特征融合层，和上采样/下采样操作。其目标是通过特征融合和上采样来增强图像的特征表达，以提高分割性能。
class DNANet(nn.Module):
    # num_blocks: 每个阶段使用的块的数量。
    # nb_filter: 网络中每个阶段的过滤器数量。
    # num_blocks: 每个阶段使用的块的数量。
    # nb_filter: 网络中每个阶段的过滤器数量。
    def __init__(self, num_classes=1,input_channels=1, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True, mode='test'):
        super(DNANet, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        # 将输入的特征图的每个 2x2 区域替换为该区域的最大值，从而减少特征图的空间尺寸。
        self.pool  = nn.MaxPool2d(2, 2)
        # 双线性插值（bilinear interpolation），这是一种常用的上采样方法，可以在放大过程中平滑图像的细节。
        # align_corners=True 是一个参数，用于控制插值时角落点的对齐方式。
        # scale_factor：将特征值放大/缩小的倍数
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        # 基础卷积层 (self.conv0_0 到 self.conv4_0):
        # 这些层定义了网络的初始卷积块，每个块逐渐增加过滤器的数量，从 nb_filter[0] 到 nb_filter[4]。
        # num_blocks 数组定义了每个阶段中卷积块的数量。
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        # 特征融合卷积层 (self.conv0_1 到 self.conv3_1):
        # 这些层通过将前一阶段的输出与当前阶段的输出在通道维度上进行拼接，实现特征融合。
        # 例如，self.conv0_1 将 self.conv0_0 和 self.conv1_0 的输出进行拼接，然后通过卷积块处理。
        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        # 进一步的特征融合卷积层 (self.conv0_2 到 self.conv1_2):
        # 这些层进一步扩展了特征融合的概念，将更多的前阶段特征图与当前阶段的特征图进行拼接。
        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        # 最终的特征融合卷积层 (self.conv0_3 和 self.conv1_3):
        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        # 最终卷积层 (self.conv0_4 和 self.conv0_4_final):将所有之前的特征图融合在一起
        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])
        # 对融合后的特征图进行进一步的卷积处理。
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        # 1x1 卷积层 (self.conv0_4_1x1 到 self.conv0_1_1x1):
        # 这些1x1卷积层用于调整通道数，以便将来自不同阶段的特征图与 self.conv0_4 的输出进行融合。
        # 1x1 卷积是一种有效的通道融合技术，可以在不改变特征图空间尺寸的情况下重新分配通道。
        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    # 用于创建多个相同的卷积块（由 block 类定义），并将它们串联成一个序列。
    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = [] # 初始化一个空列表 layers，用于存储创建的卷积块。
        layers.append(block(input_channels, output_channels)) # 将第一个卷积块添加到 layers 列表中。
        # 使用 for 循环，根据 num_blocks 的值（减去 1，因为第一个块已经添加），创建剩余的卷积块。每个新块的输入和输出通道数都是 output_channels。
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        # 逐步下采样和特征提取:
        # x1_0 通过对 x0_0 应用池化操作 self.pool 并传递到 self.conv1_0 层得到的下采样特征图。
        x1_0 = self.conv1_0(self.pool(x0_0))
        # 特征融合:
        # x0_1 是通过将 x0_0 与 x1_0 上采样的版本 self.up(x1_0) 在通道维度上进行拼接，然后传递到 self.conv0_1 层得到的特征图。
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # 多尺度特征融合:
        # 在每个递归步骤中，特征图会通过上采样和下采样与其他尺度的特征图进行融合。
        # 例如，x1_1 是将 x1_0、x2_0 上采样的版本和 x0_1 下采样的版本进行拼接和卷积操作得到的。
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1).sigmoid()
            output2 = self.final2(x0_2).sigmoid()
            output3 = self.final3(x0_3).sigmoid()
            output4 = self.final4(Final_x0_4).sigmoid()
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(Final_x0_4).sigmoid()
            return output
