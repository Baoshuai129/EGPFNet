import pywt
import torch
import torch.nn as nn
import torchvision.ops
import math
import utils.utils
import PixelAlign2 as PixelAlign2
import torch.nn.functional as F
from einops import rearrange


class Net(nn.Module):
    def __init__(self, angRes_in, angRes_out):
        super(Net, self).__init__()
        channels = 64
        n_group = 3
        n_block = 4
        self.angRes_in = angRes_in  # 2
        self.angRes_out = angRes_out  # 7
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes_in, padding=angRes_in,
                                   bias=False)

        self.DistgGroup = CascadedDistgGroup(n_group, n_block, angRes_in, channels)
        self.UpSample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=angRes_in, stride=angRes_in, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * angRes_out * angRes_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, dilation=angRes_out, padding=angRes_out, bias=False)
        )
        # self.ResidualRefinement = PixelAlign.Pixel_align(1)
        self.Pixel_align = PixelAlign2.Pixel_align(1)

    def forward(self, x):  # 前向传播
        x = SAI2MacPI(x, self.angRes_in)
        buffer = self.init_conv(x)
        buffer = self.DistgGroup(buffer)
        out = self.UpSample(buffer)

        out = self.Pixel_align(out)

        out = MacPI2SAI(out, self.angRes_out)
        return out


class CascadedDistgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):  # 4,4,2,64
        super(CascadedDistgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DistgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.fuse = nn.Conv2d(n_group * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        for i in range(self.n_group):
            x = self.Group[i](x)
            temp.append(x)
        out = torch.cat(temp, dim=1)
        return self.fuse(out)


class DistgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):  # 4,2,64
        super(DistgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DistgBlock(angRes, channels))
        self.Blocks = nn.Sequential(*Blocks)
        self.fuse = nn.Conv2d((n_block + 1) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        temp.append(x)
        for i in range(self.n_block):
            x = self.Blocks[i](x)
            temp.append(x)
        out = torch.cat(temp, dim=1)
        return self.fuse(out)


class DistgBlock(nn.Module):
    def __init__(self, angRes, channels, first=False):  # 2,64
        super(DistgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels, channels

        # 初始AFE、SFE
        self.angRes = angRes
        self.AngFE = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes)
        )

        self.SpaFE = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False))

        self.squeezeConv = nn.Sequential(
            nn.Conv2d(2 * SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(inplace=True),
        )

        self.AngConvSq = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),bias=False),
        )

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),bias=False),
        )

        self.epi_trans = BasicTrans(channels, channels * 2)

        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, x):
        # interact
        buffer_ang1 = self.AngFE(x)
        buffer_spa1 = self.SpaFE(x)

        epi_trans = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.angRes, v=self.angRes)

        [_, _, _, h, w] = epi_trans.size()

        # Horizontal
        buffer_h = rearrange(epi_trans, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer_h = self.epi_trans(buffer_h)
        buffer_h = rearrange(buffer_h, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer_h = self.conv(buffer_h) + epi_trans
        buffer_h = rearrange(buffer_h, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)

        # Vertical
        buffer_v = rearrange(epi_trans, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer_v = self.epi_trans(buffer_v)
        buffer_v = rearrange(buffer_v, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer_v = self.conv(buffer_v) + epi_trans
        buffer_v = rearrange(buffer_v, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)

        buffer_1 = torch.cat((buffer_h, buffer_v), dim=1)
        buffer_1 = self.AngConvSq(buffer_1)

        buffer_spa2 = self.SpaConv(buffer_1)  # epi----->spa
        buffer_spa2 = torch.cat((buffer_spa1, buffer_spa2), dim=1)
        buffer_spa2 = self.AngConvSq(buffer_spa2) + buffer_spa1


        # epi-------->ang
        buffer_ang2 = self.AngFE(buffer_1)
        buffer_ang2 = torch.cat((buffer_ang2, buffer_ang1), dim=1)
        buffer_ang2 = self.AngConvSq(buffer_ang2) + buffer_ang1


        buffer_3 = torch.cat((buffer_spa2, buffer_ang2), dim=1)
        buffer_3 = self.AngConvSq(buffer_3)
        buffer_4 = torch.cat((buffer_3, buffer_h), dim=1)
        buffer_4 = self.AngConvSq(buffer_4)
        buffer_5 = torch.cat((buffer_4, buffer_v), dim=1)
        buffer_5 = self.AngConvSq(buffer_5)



        buffer = torch.cat((x, buffer_spa2 , buffer_3,  buffer_4, buffer_5 ), dim=1)
        buffer = self.squeezeConv(buffer)

        y = self.SpaConv(buffer) + buffer

        return y


class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim * 2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()

        epi_token = rearrange(buffer, 'b c n v w  -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,

                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tentor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


if __name__ == "__main__":
    net = Net(angRes_in=2, angRes_out=7).cuda()
    from thop import profile

    input = torch.randn(1, 1, 128, 128).cuda()
    flops, params = profile(net, inputs=(input,))

    print('   Number of parameters: %.2fM' % (params / 1024 ** 2))
    print('   Number of FLOPs: %.2fG' % (flops / 1024 ** 3))