import torch.nn as nn
import torch
import math
import torch.nn.init as init

def default_conv(in_channels, out_channels, kernel_size, strides=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=strides,
        padding=(kernel_size // 2), bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size, strides=1, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=(strides, strides, strides),
        padding=(kernel_size // 2), bias=bias)


class conv_block_standard(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num=3):
        super(conv_block_standard, self).__init__()
        conv_list = []
        conv_list.append(default_conv(in_channels, out_channels, kernel_size=3))
        conv_list.append(nn.ReLU(inplace=True))
        for _ in range(conv_num - 1):
            conv_list.append(default_conv(out_channels, out_channels, kernel_size=3))
            conv_list.append(nn.ReLU(inplace=True))
        # conv = conv + input_layer
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x):
        return self.conv(x)


class concat_block_standard(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num=3):
        super(concat_block_standard, self).__init__()
        conv_list = []
        conv_list.append(default_conv(in_channels, in_channels // 2, kernel_size=3))
        conv_list.append(nn.ReLU(inplace=True))
        conv_list.append(default_conv(in_channels // 2, out_channels, kernel_size=3))
        conv_list.append(nn.ReLU(inplace=True))
        conv_list.append(default_conv(out_channels, out_channels, kernel_size=3))
        conv_list.append(nn.ReLU(inplace=True))
        # conv = conv + input_layer
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv=default_conv, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            # nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            conv(in_channel, out_channel, kernel_size=3, strides=strides),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            conv(out_channel, out_channel, kernel_size=3, strides=strides),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = conv(in_channel, out_channel, kernel_size=1, strides=strides)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, conv=default_conv, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 2 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Upsampler_3d_nt_inU(nn.Module):
    def __init__(self, scale, n_feat, conv=default_conv, bn=False, act=False, bias=True):
        super(Upsampler_3d_nt_inU, self).__init__()
        self.scale = scale
        if scale == 2:
            self.conv = conv(n_feat, 2 * n_feat, 3, bias)
            self.pixelshuffle = nn.PixelShuffle(2)
        elif scale == 3:
            self.conv = conv(n_feat, 9 * n_feat, 3, bias)
            self.pixelshuffle = nn.PixelShuffle(3)

    def forward(self, x):
        out = self.conv(x)
        B, C, Z, H, W = out.shape
        # protect the contiguous of nz
        # B Z C H W
        out = out.transpose(1, 2).contiguous()
        out = out.view(B * Z, C, H, W)
        out = self.pixelshuffle(out)
        # convert to BCZHW
        out = out.view(B, Z, C // (self.scale ** 2), H * self.scale, W * self.scale).transpose(1, 2).contiguous()
        return out

class Upsampler_3d_nt(nn.Module):
    def __init__(self, scale, n_feat, conv=default_conv, bn=False, act=False, bias=True):
        super(Upsampler_3d_nt, self).__init__()
        self.scale = scale
        if scale == 2:
            self.conv = conv(n_feat, 4 * n_feat, 3, bias)
            self.pixelshuffle = nn.PixelShuffle(2)
        elif scale == 3:
            self.conv = conv(n_feat, 9 * n_feat, 3, bias)
            self.pixelshuffle = nn.PixelShuffle(3)

    def forward(self, x):
        out = self.conv(x)
        B, C, Z, H, W = out.shape
        # protect the contiguous of nz
        # B Z C H W
        out = out.transpose(1, 2).contiguous()
        out = out.view(B * Z, C, H, W)
        out = self.pixelshuffle(out)
        # convert to BCZHW
        out = out.view(B, Z, C // (self.scale ** 2), H * self.scale, W * self.scale).transpose(1, 2).contiguous()
        return out

class UNet_3D(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet_3D, self).__init__()

        self.dim = dim
        # in
        self.ConvBlock1 = ConvBlock(1, dim, conv=default_conv3d)
        self.pool1 = nn.Conv3d(dim, dim, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock2 = block(dim, dim * 2, conv=default_conv3d)
        self.pool2 = nn.Conv3d(dim * 2, dim * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, conv=default_conv3d)
        self.pool3 = nn.Conv3d(dim * 4, dim * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, conv=default_conv3d)
        self.pool4 = nn.Conv3d(dim * 8, dim * 8, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, conv=default_conv3d)

        self.upv6 = Upsampler_3d_nt_inU(2, dim * 16, conv=default_conv3d)
        self.ConvBlock6 = block(dim * 16, dim * 8, conv=default_conv3d)

        self.upv7 = Upsampler_3d_nt_inU(2, dim * 8, conv=default_conv3d)
        self.ConvBlock7 = block(dim * 8, dim * 4, conv=default_conv3d)

        self.upv8 = Upsampler_3d_nt_inU(2, dim * 4, conv=default_conv3d)
        self.ConvBlock8 = block(dim * 4, dim * 2, conv=default_conv3d)

        self.upv9 = Upsampler_3d_nt_inU(2, dim * 2, conv=default_conv3d)
        self.ConvBlock9 = block(dim * 2, dim, conv=default_conv3d)

        # out
        self.out_projection = nn.Conv3d(dim, 1, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        out = self.out_projection(conv9)

        # return out + x
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,
                                                                                           nn.ConvTranspose3d) or isinstance(
                    m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)

class UNet_3D_up3(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet_3D_up3, self).__init__()

        self.dim = dim
        # in
        self.ConvBlock1 = ConvBlock(1, dim, conv=default_conv3d)
        self.pool1 = nn.Conv3d(dim, dim, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock2 = block(dim, dim * 2, conv=default_conv3d)
        self.pool2 = nn.Conv3d(dim * 2, dim * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, conv=default_conv3d)
        self.pool3 = nn.Conv3d(dim * 4, dim * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, conv=default_conv3d)
        self.pool4 = nn.Conv3d(dim * 8, dim * 8, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, conv=default_conv3d)

        self.upv6 = Upsampler_3d_nt_inU(2, dim * 16, conv=default_conv3d)
        self.ConvBlock6 = block(dim * 16, dim * 8, conv=default_conv3d)

        self.upv7 = Upsampler_3d_nt_inU(2, dim * 8, conv=default_conv3d)
        self.ConvBlock7 = block(dim * 8, dim * 4, conv=default_conv3d)

        self.upv8 = Upsampler_3d_nt_inU(2, dim * 4, conv=default_conv3d)
        self.ConvBlock8 = block(dim * 4, dim * 2, conv=default_conv3d)

        self.upv9 = Upsampler_3d_nt_inU(2, dim * 2, conv=default_conv3d)
        self.ConvBlock9 = block(dim * 2, dim, conv=default_conv3d)

        self.upv10 = Upsampler_3d_nt(3, dim, conv=default_conv3d)
        self.ConvBlock10 = block(dim, dim, conv=default_conv3d)

        # out

        self.out_projection = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, 1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
        )

        self._init_weights()

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        up10 = self.upv10(conv9)
        conv10 = self.ConvBlock10(up10)

        out = self.out_projection(conv10)

        # return out + x
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,
                                                                                           nn.ConvTranspose3d) or isinstance(
                    m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)

class UNet_3D_up2(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet_3D_up2, self).__init__()

        self.dim = dim
        # in
        self.ConvBlock1 = ConvBlock(1, dim, conv=default_conv3d)
        self.pool1 = nn.Conv3d(dim, dim, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock2 = block(dim, dim * 2, conv=default_conv3d)
        self.pool2 = nn.Conv3d(dim * 2, dim * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, conv=default_conv3d)
        self.pool3 = nn.Conv3d(dim * 4, dim * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, conv=default_conv3d)
        self.pool4 = nn.Conv3d(dim * 8, dim * 8, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, conv=default_conv3d)

        self.upv6 = Upsampler_3d_nt_inU(2, dim * 16, conv=default_conv3d)
        self.ConvBlock6 = block(dim * 16, dim * 8, conv=default_conv3d)

        self.upv7 = Upsampler_3d_nt_inU(2, dim * 8, conv=default_conv3d)
        self.ConvBlock7 = block(dim * 8, dim * 4, conv=default_conv3d)

        self.upv8 = Upsampler_3d_nt_inU(2, dim * 4, conv=default_conv3d)
        self.ConvBlock8 = block(dim * 4, dim * 2, conv=default_conv3d)

        self.upv9 = Upsampler_3d_nt_inU(2, dim * 2, conv=default_conv3d)
        self.ConvBlock9 = block(dim * 2, dim, conv=default_conv3d)

        self.upv10 = Upsampler_3d_nt(2, dim, conv=default_conv3d)
        self.ConvBlock10 = block(dim, dim, conv=default_conv3d)

        # out

        self.out_projection = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, 1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
        )

        self._init_weights()

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        up10 = self.upv10(conv9)
        conv10 = self.ConvBlock10(up10)

        out = self.out_projection(conv10)

        # return out + x
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,
                                                                                           nn.ConvTranspose3d) or isinstance(
                    m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)


"""
# --------------------------------------------
https://github.com/LabForComputationalVision/bias_free_denoising/blob/master/models/unet.py
# --------------------------------------------
"""


# --------------------------------------------
# unet in noise2noise paper
# --------------------------------------------

class UNet_3D_N2N(nn.Module):

    def __init__(self, in_nc=1, out_nc=1):
        '''
        initialize the unet
        '''
        super(UNet_3D_N2N, self).__init__()
        in_channels = in_nc
        out_channels = out_nc
        self.encode1 = nn.Sequential(
            nn.Conv3d(in_channels, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((2, 2, 2)))

        self.encode2 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((2, 2, 2)))

        self.encode3 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode4 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode5 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode6 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.ConvTranspose3d(48, 48, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode1 = nn.Sequential(
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode2 = nn.Sequential(
            nn.Conv3d(144, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode3 = nn.Sequential(
            nn.Conv3d(144, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1),
                               bias=False))

        self.decode4 = nn.Sequential(
            nn.Conv3d(144, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1),
                               bias=False))

        self.decode5 = nn.Sequential(
            nn.Conv3d(96 + in_channels, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(64, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01))

        self.output_layer = nn.Conv3d(32, out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        self._init_weights()

    def forward(self, x):  # z_size is 4
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)
        # print(pool5.shape)
        upsample5 = self.encode6(pool5)
        # print(upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode1(concat5)
        # print(upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode2(concat4)
        # print(upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode3(concat3)
        # print(upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode4(concat2)
        # print(upsample1.shape)
        concat1 = torch.cat((upsample1, x), dim=1)
        upsample0 = self.decode5(concat1)
        # print(upsample0.shape)
        output = self.output_layer(upsample0)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,
                                                                                           nn.ConvTranspose3d) or isinstance(
                    m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)


class UNet_3D_N2N_NT(nn.Module):

    def __init__(self, in_nc=1, out_nc=1):
        '''
        initialize the unet
        '''
        super(UNet_3D_N2N_NT, self).__init__()
        in_channels = in_nc
        out_channels = out_nc
        self.encode1 = nn.Sequential(
            nn.Conv3d(in_channels, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode2 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode3 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode4 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode5 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1, 2, 2)))

        self.encode6 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.ConvTranspose3d(48, 48, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode1 = nn.Sequential(
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode2 = nn.Sequential(
            nn.Conv3d(144, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode3 = nn.Sequential(
            nn.Conv3d(144, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode4 = nn.Sequential(
            nn.Conv3d(144, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),
                               bias=False))

        self.decode5 = nn.Sequential(
            nn.Conv3d(96 + in_channels, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(64, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01))

        self.output_layer = nn.Conv3d(32, out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        self._init_weights()

    def forward(self, x):  # z_size is 4
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)
        # print(pool5.shape)
        upsample5 = self.encode6(pool5)
        # print(upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode1(concat5)
        # print(upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode2(concat4)
        # print(upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode3(concat3)
        # print(upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode4(concat2)
        # print(upsample1.shape)
        concat1 = torch.cat((upsample1, x), dim=1)
        upsample0 = self.decode5(concat1)
        # print(upsample0.shape)
        output = self.output_layer(upsample0)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,
                                                                                           nn.ConvTranspose3d) or isinstance(
                    m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)

class UNet_2D_N2N(nn.Module):

    def __init__(self, in_nc=1, out_nc=1):
        '''
        initialize the unet
        '''
        super(UNet_2D_N2N, self).__init__()
        in_channels = in_nc
        out_channels = out_nc
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2)))

        self.encode2 = nn.Sequential(
            nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2)))

        self.encode3 = nn.Sequential(
            nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2)))

        self.encode4 = nn.Sequential(
            nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2)))

        self.encode5 = nn.Sequential(
            nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2)))

        self.encode6 = nn.Sequential(
            nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ConvTranspose2d(48, 48, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                               bias=False))

        self.decode1 = nn.Sequential(
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(96, 96, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                               bias=False))

        self.decode2 = nn.Sequential(
            nn.Conv2d(144, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(96, 96, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                               bias=False))

        self.decode3 = nn.Sequential(
            nn.Conv2d(144, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(96, 96, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                               bias=False))

        self.decode4 = nn.Sequential(
            nn.Conv2d(144, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(96, 96, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                               bias=False))

        self.decode5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.01))

        self.output_layer = nn.Conv2d(32, out_channels, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self._init_weights()

    def forward(self, x):  # z_size is 4
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)
        # print(pool5.shape)
        upsample5 = self.encode6(pool5)
        # print(upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode1(concat5)
        # print(upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode2(concat4)
        # print(upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode3(concat3)
        # print(upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode4(concat2)
        # print(upsample1.shape)
        concat1 = torch.cat((upsample1, x), dim=1)
        upsample0 = self.decode5(concat1)
        # print(upsample0.shape)
        output = self.output_layer(upsample0)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m,
                                                                                           nn.ConvTranspose3d) or isinstance(
                    m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)

class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)

class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Padding(nn.Module):
    def __init__(self, padding, padding_mode='zeros', value=0):
        super(Padding, self).__init__()
        if padding_mode == 'reflection':
            self. padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)

class ResBlock(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm='inorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []

        # 1st conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        # 2nd conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=[])]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
class ResNet(nn.Module):
    def __init__(self, nch_in=1, nch_out=1, nch_ker=64, norm='bnorm', nblk=16):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=[], relu=0.0)

        res = []
        for i in range(self.nblk):
            res += [ResBlock(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
        self.res = nn.Sequential(*res)

        self.dec1 = CNR2d(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=norm, relu=[])

        self.conv1 = Conv2d(self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.enc1(x)
        x0 = x

        x = self.res(x)

        x = self.dec1(x)
        x = x + x0

        x = self.conv1(x)

        return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(
                    m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UpsampleCat(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UpsampleCat, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc

        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 2, 2, 0, False)
        initialize_weights(self.deconv, 0.1)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        return torch.cat([x1, x2], dim=1)


def conv_func(x, conv, blindspot):
    size = conv.kernel_size[0]
    if blindspot:
        assert (size % 2) == 1
    ofs = 0 if (not blindspot) else size // 2

    if ofs > 0:
        # (padding_left, padding_right, padding_top, padding_bottom)
        pad = nn.ConstantPad2d(padding=(0, 0, ofs, 0), value=0)
        x = pad(x)
    x = conv(x)
    if ofs > 0:
        x = x[:, :, :-ofs, :]
    return x


def pool_func(x, pool, blindspot):
    if blindspot:
        pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
        x = pad(x[:, :, :-1, :])
    x = pool(x)
    return x


def rotate(x, angle):
    if angle == 0:
        return x
    elif angle == 90:
        return torch.rot90(x, k=1, dims=(3, 2))
    elif angle == 180:
        return torch.rot90(x, k=2, dims=(3, 2))
    elif angle == 270:
        return torch.rot90(x, k=3, dims=(3, 2))


class UNet(nn.Module):
    def __init__(self,
                 in_nc=1,
                 out_nc=1,
                 n_feature=48,
                 blindspot=False,
                 zero_last=False):
        super(UNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_feature = n_feature
        self.blindspot = blindspot
        self.zero_last = zero_last
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Encoder part
        self.enc_conv0 = nn.Conv2d(self.in_nc, self.n_feature, 3, 1, 1)
        self.enc_conv1 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv0, 0.1)
        initialize_weights(self.enc_conv1, 0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv2 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv2, 0.1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv3, 0.1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_conv4 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv4, 0.1)
        self.pool4 = nn.MaxPool2d(2)

        self.enc_conv5 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv5, 0.1)
        self.pool5 = nn.MaxPool2d(2)

        self.enc_conv6 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv6, 0.1)

        # Decoder part
        self.up5 = UpsampleCat(self.n_feature, self.n_feature)
        self.dec_conv5a = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv5b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv5a, 0.1)
        initialize_weights(self.dec_conv5b, 0.1)

        self.up4 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv4a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv4b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv4a, 0.1)
        initialize_weights(self.dec_conv4b, 0.1)

        self.up3 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv3a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv3b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv3a, 0.1)
        initialize_weights(self.dec_conv3b, 0.1)

        self.up2 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv2a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv2b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv2a, 0.1)
        initialize_weights(self.dec_conv2b, 0.1)

        self.up1 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)

        # Output stages
        self.dec_conv1a = nn.Conv2d(self.n_feature * 2 + self.in_nc, 96, 3, 1,
                                    1)
        initialize_weights(self.dec_conv1a, 0.1)
        self.dec_conv1b = nn.Conv2d(96, 96, 3, 1, 1)
        initialize_weights(self.dec_conv1b, 0.1)
        if blindspot:
            self.nin_a = nn.Conv2d(96 * 4, 96 * 4, 1, 1, 0)
            self.nin_b = nn.Conv2d(96 * 4, 96, 1, 1, 0)
        else:
            self.nin_a = nn.Conv2d(96, 96, 1, 1, 0)
            self.nin_b = nn.Conv2d(96, 96, 1, 1, 0)
        initialize_weights(self.nin_a, 0.1)
        initialize_weights(self.nin_b, 0.1)
        self.nin_c = nn.Conv2d(96, self.out_nc, 1, 1, 0)
        if not self.zero_last:
            initialize_weights(self.nin_c, 0.1)

    def forward(self, x):
        # Input stage
        blindspot = self.blindspot
        if blindspot:
            x = torch.cat([rotate(x, a) for a in [0, 90, 180, 270]], dim=0)
        # Encoder part
        pool0 = x
        x = self.act(conv_func(x, self.enc_conv0, blindspot))
        x = self.act(conv_func(x, self.enc_conv1, blindspot))
        x = pool_func(x, self.pool1, blindspot)
        pool1 = x

        x = self.act(conv_func(x, self.enc_conv2, blindspot))
        x = pool_func(x, self.pool2, blindspot)
        pool2 = x

        x = self.act(conv_func(x, self.enc_conv3, blindspot))
        x = pool_func(x, self.pool3, blindspot)
        pool3 = x

        x = self.act(conv_func(x, self.enc_conv4, blindspot))
        x = pool_func(x, self.pool4, blindspot)
        pool4 = x

        x = self.act(conv_func(x, self.enc_conv5, blindspot))
        x = pool_func(x, self.pool5, blindspot)

        x = self.act(conv_func(x, self.enc_conv6, blindspot))

        # Decoder part
        x = self.up5(x, pool4)
        x = self.act(conv_func(x, self.dec_conv5a, blindspot))
        x = self.act(conv_func(x, self.dec_conv5b, blindspot))

        x = self.up4(x, pool3)
        x = self.act(conv_func(x, self.dec_conv4a, blindspot))
        x = self.act(conv_func(x, self.dec_conv4b, blindspot))

        x = self.up3(x, pool2)
        x = self.act(conv_func(x, self.dec_conv3a, blindspot))
        x = self.act(conv_func(x, self.dec_conv3b, blindspot))

        x = self.up2(x, pool1)
        x = self.act(conv_func(x, self.dec_conv2a, blindspot))
        x = self.act(conv_func(x, self.dec_conv2b, blindspot))

        x = self.up1(x, pool0)

        # Output stage
        if blindspot:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
            x = pad(x[:, :, :-1, :])
            x = torch.split(x, split_size_or_sections=x.shape[0] // 4, dim=0)
            x = [rotate(y, a) for y, a in zip(x, [0, 270, 180, 90])]
            x = torch.cat(x, dim=1)
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        else:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        return x

if __name__ == '__main__':
    upscale = 1
    depth = 8
    height = 64
    width = 64
    model = UNet_3D().cuda().eval()

    print(height, width)

    x = torch.randn((1, 1,depth, height, width)).cuda()
    x = model(x)

    print(x.shape)
