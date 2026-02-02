import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(UNet, self).__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)  # 1 -> 64
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels*2))  # 64 -> 128
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*2, base_channels*4))  # 128 -> 256
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*4, base_channels*8))  # 256 -> 512
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*8, base_channels*8))  # 512 -> 512

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels*8, base_channels*8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels*16, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels*8, base_channels*2)

        self.up3 = nn.ConvTranspose2d(base_channels*2, base_channels*2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels*4, base_channels)

        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels*2, base_channels)

        # Output
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()   # 或 nn.Softmax(dim=1) 用于多分类分割

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)   # [bs,64,224,224]
        x2 = self.down1(x1)  # [bs,128,112,112]
        x3 = self.down2(x2)  # [bs,256,56,56]
        x4 = self.down3(x3)  # [bs,512,28,28]
        x5 = self.down4(x4)  # [bs,512,14,14]

        # Decoder
        x = self.up1(x5)  # [bs,512,28,28]
        x = self.conv1(torch.cat([x, x4], dim=1))

        x = self.up2(x)  # [bs,256,56,56]
        x = self.conv2(torch.cat([x, x3], dim=1))

        x = self.up3(x)  # [bs,128,112,112]
        x = self.conv3(torch.cat([x, x2], dim=1))

        x = self.up4(x)  # [bs,64,224,224]
        x = self.conv4(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return self.act(logits)


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1).to('cuda')
    x = torch.randn(2, 1, 224, 224).to('cuda')
    labels = torch.tensor([0, 1]).to('cuda')
    out = model(x)
    print(f'Out.shape: {out.shape}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
    print(f'Range: {out.min().item()} ~ {out.max().item()}')