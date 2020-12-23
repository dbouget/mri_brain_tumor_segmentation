import torch.nn as nn
import torch
import torch.utils.checkpoint as cp


class DSConv3D(nn.Module):
    def __init__(self, in_chans, out_chans, dilation=1, dstride=2, padding=1):
        super(DSConv3D, self).__init__()
        self.dConv = nn.Conv3d(in_chans, in_chans, kernel_size=3, stride=dstride, padding=padding,
                               dilation=dilation, groups=in_chans, bias=False)
        self.conv = nn.Conv3d(in_chans, out_chans, kernel_size=1, dilation=1, stride=1, bias=False)
        self.norm = nn.BatchNorm3d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        #self.avgpool = nn.AvgPool3d(kernel_size=3, stride=dstride)

    def forward(self, x):
        out = self.dConv(x)
        out = self.conv(out)
        out = self.relu(self.norm(out))
        #out = self.avgpool(out)
        return out


class DrdbBlock3D(nn.Module):
    def __init__(self, in_chans, out_chans, growth_rate, nr_blocks=4):
        super(DrdbBlock3D, self).__init__()
        self.nr_blocks = nr_blocks
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.growth_rate = growth_rate
        self.memory_efficient = True

        self.ds_conv_1 = DSConv3D(in_chans=self.in_chans, out_chans=growth_rate, dilation=1, dstride=1,
                                  padding=1)
        self.ds_conv_2 = DSConv3D(in_chans=self.in_chans + growth_rate, out_chans=growth_rate, dilation=2,
                                  dstride=1, padding=2)
        self.ds_conv_3 = DSConv3D(in_chans=self.in_chans + growth_rate * 2, out_chans=growth_rate,
                                  dilation=3, dstride=1, padding=3)
        self.ds_conv_4 = DSConv3D(in_chans=self.in_chans + growth_rate * 3, out_chans=growth_rate,
                                  dilation=4, dstride=1, padding=4)

        self.conv = nn.Conv3d(in_chans + growth_rate * 4, self.out_chans, kernel_size=1)
        #self.avgpool = nn.AvgPool3d(kernel_size=3)

    def forward(self, x):
        if self.memory_efficient:
            out = cp.checkpoint(self.bottleneck_function, x)
        else:
            out = self.bottleneck_function(x)
        #out = self.avgpool(out)
        return out

    def bottleneck_function(self, x):
        out = self.ds_conv_1(x)
        cat = torch.cat([out, x], 1)
        out = self.ds_conv_2(cat)
        cat = torch.cat([out, cat], 1)
        out = self.ds_conv_3(cat)
        cat = torch.cat([out, cat], 1)
        out = self.ds_conv_4(cat)
        cat = torch.cat([out, cat], 1)
        out = self.conv(cat)
        out = torch.add(out, x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(DecoderBlock, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.ds_conv = DSConv3D(in_chans=in_chans, out_chans=out_chans, dilation=1, dstride=1)
        self.upsampled = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        out = self.ds_conv(x)
        out = self.upsampled(out)
        return out
