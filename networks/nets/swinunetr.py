import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR as sur

class SwinUNETR(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = sur(**kwargs)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.act(x)

if __name__ == "__main__":
    # x = torch.randn(2, 1, 64, 64, 64).to('cuda')
    # model = SwinUNETR(
    #     img_size=(64, 64, 64),
    #     in_channels=1,
    #     out_channels=1,
    #     feature_size=48,
    #     use_checkpoint=True,
    # ).to('cuda')
    # out = model(x)
    # print(out.shape)  # torch.Size([2, 1, 64, 64, 64])
    # print(model)
    # print('='*30)
    
    x = torch.randn(2, 1, 224, 224).to('cuda')
    model = SwinUNETR(
        img_size=(224, 224),
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=2,
    ).to('cuda')
    out = model(x)
    print(out.shape)  # torch.Size([2, 1, 224, 224])
    print(model)