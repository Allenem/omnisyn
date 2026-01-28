import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super(UNetEncoder, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

    def forward(self, x):
        x1 = self.inc(x)   # [bs,64,224,224]
        x2 = self.down1(x1) # [bs,128,112,112]
        x3 = self.down2(x2) # [bs,256,56,56]
        x4 = self.down3(x3) # [bs,512,28,28]
        x5 = self.down4(x4) # [bs,512,14,14]
        return x1, x2, x3, x4, x5


class UNetDecoderNoCond(nn.Module):
    """不带 condition 的 decoder"""
    def __init__(self, out_channels=1):
        super(UNetDecoderNoCond, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv1 = DoubleConv(512 + 512, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv2 = DoubleConv(256 + 256, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv3 = DoubleConv(128 + 128, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv4 = DoubleConv(64 + 64, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        logits = self.activation(logits)
        return logits


class UNetDecoderCond(nn.Module):
    """带 condition 的 decoder"""
    def __init__(self, out_channels=1, label_emb_dim=16, num_labels=3):
        super(UNetDecoderCond, self).__init__()
        self.label_emb = nn.Embedding(num_labels, label_emb_dim)

        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv1 = DoubleConv(512 + 512 + 1, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv2 = DoubleConv(256 + 256 + 1, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv3 = DoubleConv(128 + 128 + 1, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv4 = DoubleConv(64 + 64 + 1, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)

    def forward(self, features, labels):
        """
        features: (x1, x2, x3, x4, x5) from encoder
        labels: [bs]
        """
        x1, x2, x3, x4, x5 = features
        bs = x1.size(0)
        label_feat = self.label_emb(labels)  # [bs, emb_dim]
        lbl0 = label_feat.unsqueeze(-1).unsqueeze(-1)  # [bs, emb_dim, 1, 1]
        lbl0 = lbl0.mean(1, keepdim=True)              # [bs,1,1,1]

        x = self.up1(x5)  # [bs,512,28,28]
        lbl1 = lbl0.expand(-1, 1, x.size(2), x.size(3))
        x = self.conv1(torch.cat([x, x4, lbl1], dim=1))

        x = self.up2(x)  # [bs,256,56,56]
        lbl2 = lbl0.expand(-1, 1, x.size(2), x.size(3))
        x = self.conv2(torch.cat([x, x3, lbl2], dim=1))

        x = self.up3(x)  # [bs,128,112,112]
        lbl3 = lbl0.expand(-1, 1, x.size(2), x.size(3))
        x = self.conv3(torch.cat([x, x2, lbl3], dim=1))

        x = self.up4(x)  # [bs,64,224,224]
        lbl4 = lbl0.expand(-1, 1, x.size(2), x.size(3))
        x = self.conv4(torch.cat([x, x1, lbl4], dim=1))

        logits = self.outc(x)
        logits = self.activation(logits)
        return logits


class UNetNoCond(nn.Module):
    """组合模型，只含不带 condition 的 decoder"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetNoCond, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels)
        self.decoder_nocond = UNetDecoderNoCond(out_channels=out_channels)

    def forward(self, x):
        features = self.encoder(x)
        out_nocond = self.decoder_nocond(features)
        return out_nocond
   

class UNetCond(nn.Module):
    """组合模型，只含带 condition 的 decoder"""
    def __init__(self, in_channels=1, out_channels=1, label_emb_dim=16, num_labels=3):
        super(UNetCond, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels)
        self.decoder_cond = UNetDecoderCond(out_channels=out_channels,
                                            label_emb_dim=label_emb_dim,
                                            num_labels=num_labels)

    def forward(self, x, labels):
        features = self.encoder(x)
        out_cond = self.decoder_cond(features, labels)
        return out_cond


if __name__ == "__main__":
    encoder = UNetEncoder(in_channels=1)
    decoder_nocond = UNetDecoderNoCond(out_channels=1)
    decoder_cond = UNetDecoderCond(out_channels=1, label_emb_dim=16, num_labels=3)

    # 输入
    x = torch.randn(2, 1, 224, 224)
    labels = torch.tensor([0, 1])

    # 单独跑 encoder
    features = encoder(x)
    print(f'Encoder feature shapes: {[f.shape for f in features]}')
    # Encoder feature shapes: [
    # torch.Size([2, 64, 224, 224]), 
    # torch.Size([2, 128, 112, 112]), 
    # torch.Size([2, 256, 56, 56]), 
    # torch.Size([2, 512, 28, 28]), 
    # torch.Size([2, 512, 14, 14])
    # ]

    # 单独跑 no-condition decoder
    out_nc = decoder_nocond(features)
    print(f'DecoderNoCond out.shape: {out_nc.shape}') # torch.Size([2, 1, 224, 224])

    # 单独跑 condition decoder
    out_c = decoder_cond(features, labels)
    print(f'DecoderCond out.shape: {out_c.shape}') # torch.Size([2, 1, 224, 224])

    ############################

    # 端到端：同时输出两个 decoder 的结果
    model1 = UNetNoCond(in_channels=1, out_channels=1)
    out_nc1 = model1(x)
    print(f'Full Model out_nocond.shape: {out_nc1.shape}') # torch.Size([2, 1, 224, 224])

    model2 = UNetCond(in_channels=1, out_channels=1, label_emb_dim=16, num_labels=3)
    out_c2 = model2(x, labels)
    print(f'Full Model out_cond.shape: {out_c2.shape}') # torch.Size([2, 1, 224, 224])
