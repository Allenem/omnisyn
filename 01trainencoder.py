# load data, load encoder model, then train encoder using contrastive learning
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import MyDataset
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.modules.swinunetr_core import SwinUNETREncoder
from torch.utils.tensorboard import SummaryWriter
from loss import ContrastiveLoss

# Dataset
train_dir = "imgs/train/"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
bs = 24
dataset = MyDataset(train_dir, transform=transform)
# train_dataset and validation dataset 8:2 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
print(f"Train dataset size: {len(train_dataset)} samples")
print(f"Validation dataset size: {len(val_dataset)} samples")
print(f"Train dataloader size: {len(train_dataloader)} batches")
print(f"Validation dataloader size: {len(val_dataloader)} batches")

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swinunetr = SwinUNETREncoder(
    img_size=(224, 224),
    in_channels=1,
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=2,
).to(device)

swinunetrv2 = SwinUNETREncoder(
    img_size=(224, 224),
    in_channels=1,
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=2,
    use_v2=True,
).to(device)

encoder = swinunetr

# Loss and Optimizer
criterion = ContrastiveLoss(temperature=0.07, mae_weight=1.0, nce_weight=1.0).to(device)
# criterion = ContrastiveLoss(temperature=0.07, mae_weight=1.0, nce_weight=2.0).to(device)
# criterion = ContrastiveLoss(temperature=0.07, mae_weight=1.0, nce_weight=0.5).to(device)
# criterion = ContrastiveLoss(temperature=0.07, mae_weight=1.0, nce_weight=0.2).to(device)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-5)
log_dir = "./logs/encoder1"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
log_file = os.path.join(log_dir, "log.txt")

num_epochs = 100
best_val_loss = float("inf")
best_epoch = -1
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    
    # Training Loop
    encoder.train()
    total_loss_train = 0.0
    mae_loss_train = 0.0
    nce_loss_train = 0.0
    for batch in train_dataloader:
        t1 = batch["T1"].to(device)
        t1c = batch["T1c"].to(device)
        t2 = batch["T2"].to(device)
        # # torch.Size([8, 1, 224, 224]) torch.Size([8, 1, 224, 224]) torch.Size([8, 1, 224, 224])
        # print(t1.shape, t1c.shape, t2.shape)

        bsz = t1.size(0)

        feats_t1 = encoder(t1)   # 6 levels of feature maps
        feats_t1c = encoder(t1c)
        feats_t2 = encoder(t2)
        
        # torch.Size([8, 48, 224, 224])
        # torch.Size([8, 48, 112, 112])
        # torch.Size([8, 96, 56, 56])
        # torch.Size([8, 192, 28, 28])
        # torch.Size([8, 384, 14, 14])
        # torch.Size([8, 768, 7, 7])
        # for f1 in feats_t1:
        #     print(f1.shape)
        # for f2 in feats_t1c:
        #     print(f2.shape)
        # for f3 in feats_t2:
        #     print(f3.shape)

        # keep last 4 feature maps
        f1 = feats_t1[-4:]
        f2 = feats_t1c[-4:]
        f3 = feats_t2[-4:]

        # last 4-2 layers: MAE loss
        # last 1 layer: InfoNCE loss
        loss = criterion(f1, f2, f3)
        total_loss, mae_loss, nce_loss = loss['total'], loss['mae'], loss['nce']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_train += total_loss.item() * bsz
        mae_loss_train += mae_loss.item() * bsz
        nce_loss_train += nce_loss.item() * bsz

        # break # only one batch for training debugging

    # break
    avg_train_loss = total_loss_train / len(train_dataloader.dataset)
    avg_mae_loss = mae_loss_train / len(train_dataloader.dataset)
    avg_nce_loss = nce_loss_train / len(train_dataloader.dataset)

    # Validation Loop
    encoder.eval()
    total_loss_val = 0.0
    mae_loss_val = 0.0
    nce_loss_val = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            t1 = batch["T1"].to(device)
            t1c = batch["T1c"].to(device)
            t2 = batch["T2"].to(device)

            bsz = t1.size(0)

            feats_t1 = encoder(t1)
            feats_t1c = encoder(t1c)
            feats_t2 = encoder(t2)
            
            f1 = feats_t1[-4:]
            f2 = feats_t1c[-4:]
            f3 = feats_t2[-4:]

            loss = criterion(f1, f2, f3)
            total_loss, mae_loss, nce_loss = loss['total'], loss['mae'], loss['nce']

            total_loss_val += total_loss.item() * bsz
            mae_loss_val += mae_loss.item() * bsz
            nce_loss_val += nce_loss.item() * bsz

            # tensorboard add hot images for first batch only, each 10 epochs
            if (epoch == 0 or epoch == num_epochs - 1 or (epoch+1) % 10 == 0) and batch_idx == 0:

                # makegrid for 3 modalities
                # print(t1.shape, t1c.shape, t2.shape) # [bs, 1, 224, 224]
                t1t1ct2 = make_grid(
                    torch.cat([t1[:4], t1c[:4], t2[:4]], dim=0),
                    nrow=4,
                    padding=2,
                    normalize=True, # 强烈建议（MRI 强度范围不统一）
                    scale_each=True
                    )
                # print(t1t1ct2.shape) # [row=12//4, H=12//4*224+2*(3+1)=680, W=4*224+2*(4+1)=906]

                # print(f1[-2].shape, f2[-2].shape, f3[-2].shape)
                # f?[-2]: [bs, 384, 14, 14] mean-> [bs, 1, 14, 14]
                hotf1 = f1[-2].detach().cpu().mean(dim=1, keepdim=True)
                hotf2 = f2[-2].detach().cpu().mean(dim=1, keepdim=True)
                hotf3 = f3[-2].detach().cpu().mean(dim=1, keepdim=True)
                # 3 行 × 4 列
                hotmapt1t1ct2 = make_grid(
                    torch.cat([hotf1[:4], hotf2[:4], hotf3[:4]], dim=0),
                    nrow=4,
                    padding=2,
                    normalize=True,     # ✔ make_grid 内部做归一化
                    scale_each=True     # ✔ 每张图各自归一化（非常关键）
                )
                # print(hotmapt1t1ct2.shape) # [row=12//4, H=3*14+2*(3+1)=50, W=4*14+2*(4+1)=66]

                writer.add_image("Val/T1T1cT2", t1t1ct2, epoch)
                writer.add_image("Val/Hotmap", hotmapt1t1ct2*255, epoch)

            # break  # only one batch for validation debugging

    avg_val_loss = total_loss_val / len(val_dataloader.dataset)
    avg_mae_val_loss = mae_loss_val / len(val_dataloader.dataset)
    avg_nce_val_loss = nce_loss_val / len(val_dataloader.dataset)

    # Log to TensorBoard
    writer.add_scalars("Loss/Total_Loss", {
        "Train": avg_train_loss,
        "Validation": avg_val_loss,
    }, epoch + 1)
    writer.add_scalars("Loss/MAE_Loss", {
        "Train": avg_mae_loss,
        "Validation": avg_mae_val_loss,
    }, epoch + 1)
    writer.add_scalars("Loss/InfoNCE_Loss", {
        "Train": avg_nce_loss,
        "Validation": avg_nce_val_loss,
    }, epoch + 1)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(encoder.state_dict(), os.path.join(log_dir, "best_encoder.pth"))
        best_epoch = epoch + 1

    # Log to txt file
    with open(log_file, "a") as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}],\n"
                f"Train Loss: {avg_train_loss:.4f} (MAE: {avg_mae_loss:.4f}, NCE: {avg_nce_loss:.4f}),\n"
                f"Validation Loss: {avg_val_loss:.4f} (MAE: {avg_mae_val_loss:.4f}, NCE: {avg_nce_val_loss:.4f})\n"
                f"Best Epoch: {best_epoch}\n\n")
        
    # break