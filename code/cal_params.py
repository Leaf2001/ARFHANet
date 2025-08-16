import argparse
from net import Net
import os
from thop import profile
import torch
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 定义时间戳
timestamp = (time.ctime()).replace(' ', '_').replace(':', '_')

# 设置参数解析器
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD Parameter and FLOPs")
parser.add_argument("--model_names", default=['Fourmodule_low'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet',"
                         "'threeConvs_CSFA_UNet','tConvs_feature','Fourmodule','Fourmodule_low','tConvs_feature_exp2','Fourmodule_low_exp2'")
global opt
opt = parser.parse_args()

# 定义全局文件
opt.f = open(f'./params_{timestamp}.txt', 'w')

if __name__ == '__main__':
    # 使用全局定义的 opt.f，无需重新打开文件
    input_img = torch.rand(1, 1, 256, 256).cuda()
    for model_name in opt.model_names:
        net = Net(model_name, mode='test').cuda()
        flops, params = profile(net, inputs=(input_img,))
        print(model_name)
        print('Params: %2fM' % (params / 1e6))
        print('FLOPs: %2fGFLOPs' % (flops / 1e9))
        opt.f.write(model_name + '\n')
        opt.f.write('Params: %2fM\n' % (params / 1e6))
        opt.f.write('FLOPs: %2fGFLOPs\n' % (flops / 1e9))
        opt.f.write('\n')
    opt.f.close()

