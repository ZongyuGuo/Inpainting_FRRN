import torch
import torch.nn as nn
from partialconv2d import PartialConv2d

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class FRRNet(BaseNetwork):
    def __init__(self, block_num=16, init_weights=True):
        super(FRRNet, self).__init__()
        self.block_num = block_num
        self.dilation_num = block_num // 2
        blocks = []
        for idx in range(self.block_num):
            blocks.append(FRRBlock())
        self.blocks = nn.ModuleList(blocks)

        if init_weights:
            self.init_weights()

    def forward(self, x, mask):
        mid_x = []
        mid_mask = []

        mask_update = mask
        for dilation_idx in range(self.dilation_num):
            x, _ = self.blocks[dilation_idx * 2](x, mask_update, mask)
            x, mask_update = self.blocks[dilation_idx * 2 + 1](x, mask_update, mask)
            mid_x.append(x)
            mid_mask.append(mask_update)

        return x, mid_x, mid_mask


class FRRBlock(nn.Module):
    def __init__(self):
        super(FRRBlock, self).__init__()
        self.full_conv1 = PartialConvLayer(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, use_norm=False)
        self.full_conv2 = PartialConvLayer(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, use_norm=False)
        self.full_conv3 = PartialConvLayer(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2, use_norm=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.branch_conv1 = PartialConvLayer(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, use_norm=False)
        self.branch_conv2 = PartialConvLayer(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1)
        self.branch_conv3 = PartialConvLayer(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.branch_conv4 = PartialConvLayer(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1, act='LeakyReLU')
        self.branch_conv5 = PartialConvLayer(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1, act='LeakyReLU')
        self.branch_conv6 = PartialConvLayer(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, act='Tanh')

    def forward(self, input, mask, mask_ori):
        x = input
        out_full, mask_full = self.full_conv1(x, mask)
        out_full, mask_full = self.full_conv2(out_full, mask_full)
        out_full, mask_full = self.full_conv3(out_full, mask_full)

        out_branch, mask_branch = self.branch_conv1(x, mask)
        out_branch, mask_branch = self.branch_conv2(out_branch, mask_branch)
        out_branch, mask_branch = self.branch_conv3(out_branch, mask_branch)

        out_branch = self.upsample(out_branch)
        mask_branch = self.upsample(mask_branch)
        out_branch, mask_branch = self.branch_conv4(out_branch, mask_branch)
        out_branch = self.upsample(out_branch)
        mask_branch = self.upsample(mask_branch)
        out_branch, mask_branch = self.branch_conv5(out_branch, mask_branch)
        out_branch = self.upsample(out_branch)
        mask_branch = self.upsample(mask_branch)
        out_branch, mask_branch = self.branch_conv6(out_branch, mask_branch)

        mask_update = mask_full * mask_branch
        out = (out_full * mask_update + out_branch * mask_update) / 2 * (1 - mask_ori) + input
        #out = (out_full * mask_update + out_branch * mask_update) / 2 * (1 - mask_ori) + input * mask_ori
        return out, mask_update



class PartialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act='ReLU', use_norm=True):
        super(PartialConvLayer, self).__init__()
        self.conv = PartialConv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride=stride, padding=padding, return_mask=True)
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        self.use_norm = use_norm
        if act == 'ReLU':
            self.act = nn.ReLU(True)
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU(0.2, True)
        elif act == 'Tanh':
            self.act = nn.Tanh()

    def forward(self, x, mask):
        x, mask_update = self.conv(x, mask)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)
        return x, mask_update



class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
