import os
import itertools
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from datetime import datetime
from loss import CLIPLoss
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer
from dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.modules.swinunetr_core import SwinUNETR_LabelCond


# ==========================
# Utils
# ==========================

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")
    f = open(log_path, "a")
    f.write(f"\n===== New Run: {datetime.now()} =====\n")
    f.flush()
    return f

def log_print(fp, msg):
    print(msg)
    fp.write(msg + "\n")
    fp.flush()

def only_log(fp, msg):
    fp.write(msg + "\n")
    fp.flush()


# ==========================
# Training Loop
# ==========================

def train_decoder(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    mse_loss,
    clip_loss,
    txt_feat, # [6, 512]
    num_epochs=200,
    log_dir="./logs",
    val_interval=10,
    lambda1=1,
    lambda2=1,
):
    writer = SummaryWriter(log_dir)
    logger = setup_logger(log_dir)

    model.train()
    model.encoder.eval()  # encoder frozen
    for p in model.encoder.parameters():
        p.requires_grad = False

    # -------- modality combinations (C32 = 6) --------
    modalities = ["T1", "T1c", "T2"]
    combs = list(itertools.permutations(modalities, 2))
    # [('T1', 'T1c'), ('T1', 'T2'), ('T1c', 'T1'), ('T1c', 'T2'), ('T2', 'T1'), ('T2', 'T1c')]

    best_val_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        # log_print(logger, f"Epoch [{epoch+1}/{num_epochs}]")

        # ================= Training =================
        for batch in train_dataloader:
            # t1  = batch["T1"].to(device)
            # t1c = batch["T1c"].to(device)
            # t2  = batch["T2"].to(device)

            for comb_id, comb in enumerate(combs):
                # ---- build input / output ----
                img_src = batch[comb[0]].to(device)
                img_tgt = batch[comb[1]].to(device)
                lbl_idx = modalities.index(comb[1])
                bsi = img_src.size(0)
                # lbl_tgt: [6, 512] -> [512] -> [B, 512]
                lbl_tgt = txt_feat[lbl_idx].unsqueeze(0).repeat(bsi, 1).to(device)
                # lbl_idx: number -> [bsi,] tensor int
                lbl_idx = torch.full((bsi,), lbl_idx, dtype=torch.long, device=device)

                # ---- forward ----
                # print(img_src.shape, lbl_idx.shape) # [4, 1, 224, 224] [24]
                pred = model(img_src, lbl_idx)

                # ---- losses ----
                loss_mse = mse_loss(pred, img_tgt)
                loss_clip = clip_loss(pred, lbl_tgt)
                loss = lambda1 * loss_mse + lambda2 * loss_clip

                # ---- backward ----
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ---- TensorBoard ----
                writer.add_scalars(f"comb_{comb_id}/MSE", {"Train": loss_mse.item()}, epoch)
                writer.add_scalars(f"comb_{comb_id}/CLIP", {"Train": loss_clip.item()}, epoch)
                writer.add_scalars(f"comb_{comb_id}/Total_Loss", {"Train": loss.item()}, epoch)

                # ---- log only ----
                only_log(logger, f"  Epoch {epoch+1}, Comb {comb_id}: MSE={loss_mse.item():.4f}, CLIP={loss_clip.item():.4f}, Total={loss.item():.4f}")

        # ================= Validation =================
        if epoch == 0 or (epoch + 1) % val_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    # t1  = batch["T1"].to(device)
                    # t1c = batch["T1c"].to(device)
                    # t2  = batch["T2"].to(device)

                    for comb_id, comb in enumerate(combs):
                        # ---- build input / output ----
                        img_src = val_batch[comb[0]].to(device)
                        img_tgt = val_batch[comb[1]].to(device)
                        lbl_idx = modalities.index(comb[1])
                        bsi = img_src.size(0)
                        # lbl_tgt: [6, 512] -> [512] -> [B, 512]
                        lbl_tgt = txt_feat[lbl_idx].unsqueeze(0).repeat(bsi, 1).to(device)
                        # lbl_idx: number -> [bsi,] tensor int
                        lbl_idx = torch.full((bsi,), lbl_idx, dtype=torch.long, device=device)

                        # ---- forward ----
                        pred = model(img_src, lbl_idx)

                        # ---- losses ----
                        loss_mse = mse_loss(pred, img_tgt)
                        loss_clip = clip_loss(pred, lbl_tgt)
                        loss = lambda1 * loss_mse + lambda2 * loss_clip

                        # ---- TensorBoard ----
                        writer.add_scalars(f"comb_{comb_id}/MSE", {"Val": loss_mse.item()}, epoch)
                        writer.add_scalars(f"comb_{comb_id}/CLIP", {"Val": loss_clip.item()}, epoch)
                        writer.add_scalars(f"comb_{comb_id}/Total_Loss", {"Val": loss.item()}, epoch)
                        grid = make_grid(torch.cat([img_src[:4], img_tgt[:4], pred[:4]], dim=0), nrow=4)
                        writer.add_image(f"comb_{comb_id}/Images", grid, epoch)

                        # ---- log only ----
                        only_log(logger, f"  [Val] Epoch {epoch+1}, Comb {comb_id}: MSE={loss_mse.item():.4f}, CLIP={loss_clip.item():.4f}, Total={loss.item():.4f}")

                        # ---- save best model ----
                        if loss.item() < best_val_loss:
                            best_val_loss = loss.item()
                            save_path = os.path.join(log_dir, "best_decoder.pth")
                            torch.save(model.state_dict(), save_path)
                            only_log(logger, f"  Best model saved at epoch {epoch+1}, comb {comb_id} with val loss {best_val_loss:.4f}")

                    #     break # only one combination
                    # break # only one batch
            model.train()

        # log_print(logger, f"Epoch {epoch+1} finished")
        # break  # only one epoch for debugging

    logger.close()
    writer.close()


# MAIN ==============================================================================


# ==========================
# 1. CLIP Related
# ==========================
clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# load pretrained weights
clipmodel.load_state_dict(torch.load("logs/clip_model_img.pth", map_location="cpu", weights_only=True))
clipmodel.eval()
# freeze CLIP model
for p in clipmodel.parameters():
    p.requires_grad = False

text_data = [
    'T1-weighted (T1) images provide high-resolution anatomical detail, with fat appearing bright and water appearing dark, useful for visualizing normal tissue structure.',
    'T1 contrast-enhanced (T1c) images involve the administration of a contrast agent, enhancing vascular structures and providing better visualization of tumors and lesions.',
    'T2-weighted (T2) images emphasize fluid-rich tissues, with water appearing bright and fat darker, making it ideal for detecting abnormalities like edema or inflammation.',
    'T2 Fluid-Attenuated Inversion Recovery MRI (FLAIR) suppresses cerebrospinal fluid (CSF) signals to better visualize pathological tissues with high water content, such as edema, tumors, or white matter lesions.',
    'Proton density (PD) weighted MRI image highlights tissues with high hydrogen atom concentration, appearing brightest in areas like fat and fluid, while minimizing T1/T2 relaxation effects for enhanced tissue contrast.',
    'Magnetic Resonance Angiography (MRA) non-invasively images blood vessels by detecting flowing blood signals, aiding in diagnosing vascular abnormalities like stenosis, aneurysms, or malformations.',
]

text_inputs = tokenizer(text_data, padding=True, return_tensors="pt", truncation=True, max_length=77)
# print("Text inputs shape:", text_inputs.input_ids.shape) # [6, 51]
# print("Text inputs:", text_inputs.input_ids)

text_outputs = clipmodel.get_text_features(input_ids=text_inputs.input_ids)
# print("Text features shape:", text_outputs.shape) # [6, 512]
# print("Text features:", text_outputs)


# ==========================
# 2. DataLoader, Model, Optimizer
# ==========================

# Dataset
train_dir = "imgs/train/"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
bs = 32
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
lambda1, lambda2 = 3, 1

model = SwinUNETR_LabelCond(
    img_size=(224, 224),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=2,
).to(device)

modelv2 = SwinUNETR_LabelCond(
    img_size=(224, 224),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=2,
    use_v2=True,
).to(device)

# load weights
model.encoder.load_state_dict(torch.load("logs/encoder1_bs24/best_encoder.pth"))
model.encoder.eval()  # encoder frozen
for p in model.encoder.parameters():
    p.requires_grad = False

# modelv2.encoder.load_state_dict(torch.load("path_to_pretrained_encoder_v2.pth"))
# modelv2.encoder.eval()  # encoder frozen
# for p in modelv2.encoder.parameters():
#     p.requires_grad = False

# Optimizer
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=1e-5)
# optimizerv2 = torch.optim.AdamW(modelv2.decoder.parameters(), lr=1e-4, weight_decay=1e-5)
mse_loss = nn.MSELoss()
clip_loss = CLIPLoss(clipmodel).to(device)


# ==========================
# 3. Train Decoder
# ==========================

train_decoder(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    device=device,
    mse_loss=mse_loss,
    clip_loss=clip_loss,
    txt_feat=text_outputs, # [6, 512]
    num_epochs=200,
    log_dir="./logs/decoder1",
    val_interval=10,
    lambda1=lambda1,
    lambda2=lambda2
)
