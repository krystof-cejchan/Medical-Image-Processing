import torch
import torch.nn as nn
from utils import UNET_BASE, UNET_BASE_CHANNEL_IN, UNET_BASE_CHANNEL_OUT


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    def __init__(self, base=UNET_BASE, in_ch=UNET_BASE_CHANNEL_IN, out_ch=UNET_BASE_CHANNEL_OUT):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(in_ch, base)             # layer1    256×256
        self.enc2 = DoubleConv(base, base * 2)          # layer2    128×128
        self.enc3 = DoubleConv(base * 2, base * 4)      # layer3    64×64
        self.enc4 = DoubleConv(base * 4, base * 8)      # layer4    32×32
        self.enc5 = DoubleConv(base * 8, base * 16)     # layer5    16×16
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bott = DoubleConv(base * 16, base * 32)    # 8×8

        # decoder
        self.up5 = nn.ConvTranspose2d(base * 32, base * 16, kernel_size = 2, stride=2)
        self.dec5 = DoubleConv(base * 32, base * 16) # l5

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size = 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8) # l4

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size = 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)  # l3

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size = 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)  # l2

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size = 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)  # l1

        # output
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # bottleneck
        b = self.bott(self.pool(e5))

        # decoder
        d5 = self.up5(b)
        d5 = self.dec5(torch.cat([d5, e5], dim=1))

        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.outc(d1)

