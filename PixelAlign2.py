import torch
import torch.nn as nn



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Pixel_align(nn.Module):
    def __init__(self, channels):
        super(Pixel_align, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        )

        self.branch_1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 5), stride=1, padding=(1, 2), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=1, padding=(2, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

        # self.conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        self.spatial_attention = SpatialAttention()

    def forward(self, x_init):


        x = self.conv_0(x_init) + x_init

        out1 = self.conv_0(x) + x

        # x_1 = self.branch_1(out1)
        # x_2 = self.branch_2(out1)
        # out2 = self.fuse( torch.cat([x_1, x_2], dim=1))


        out = self.conv_0(out1) + out1



        return out + x
