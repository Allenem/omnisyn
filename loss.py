import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMapMAELoss(nn.Module):
    """
    MAE alignment loss for feature maps
    Only aligns features within the same slice (same batch index)
    """

    def __init__(self):
        super().__init__()

    def forward(self, f_t1, f_t1c, f_t2):
        """
        Args:
            f_t1, f_t1c, f_t2: [B, C, H, W]
        """
        loss_t1_t1c = F.l1_loss(f_t1, f_t1c, reduction="mean")
        loss_t1_t2  = F.l1_loss(f_t1, f_t2,  reduction="mean")
        loss_t1c_t2 = F.l1_loss(f_t1c, f_t2, reduction="mean")

        loss = (loss_t1_t1c + loss_t1_t2 + loss_t1c_t2) / 3.0
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for slice-level representations
    Each slice has 3 views (T1 / T1c / T2)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats):
        """
        feats: [3B, D]
        """
        feats = F.normalize(feats, dim=1)

        # cosine similarity matrix
        sim = torch.matmul(feats, feats.T) / self.temperature
        sim.fill_diagonal_(-1e9)

        N = feats.shape[0]  # 3B
        labels = torch.arange(N, device=feats.device) // 3
        # index:                      0 1 2 | 3 4 5 | 6 7 8 | ...
        # labels(indicate the slice): 0 0 0 | 1 1 1 | 2 2 2 | ...

        loss = 0.0
        for i in range(N):
            pos = labels == labels[i]
            pos[i] = False
            # i = 0:
            # pos = [False, True, True, | False, False, False, | False, False, False, ...]

            numerator = torch.logsumexp(sim[i][pos], dim=0)
            denominator = torch.logsumexp(sim[i], dim=0)
            loss += -(numerator - denominator)

        return loss / N


class ContrastiveLoss(nn.Module):
    """
    Multi-level feature alignment:
    - Shallow & mid layers: MAE feature map alignment
    - Deep layer: InfoNCE on slice-level representations
    """

    def __init__(self, temperature=0.07, mae_weight=1.0, nce_weight=0.5):
        super().__init__()
        self.mae_loss = FeatureMapMAELoss()
        self.nce_loss = InfoNCELoss(temperature)

        self.mae_weight = mae_weight
        self.nce_weight = nce_weight

    def forward(self, feats_t1, feats_t1c, feats_t2):
        """
        Args:
            feats_t1, feats_t1c, feats_t2:
                list of feature maps from different levels
        """
        assert len(feats_t1) == len(feats_t1c) == len(feats_t2)

        # -------- Shallow & Mid levels: MAE --------
        num_levels = len(feats_t1)
        mae_total = 0.0
        for i in range(num_levels - 1):
            mae_total += self.mae_loss(
                feats_t1[i],
                feats_t1c[i],
                feats_t2[i]
            ) # each input: [B,C,H,W]
        mae_loss = mae_total / (num_levels - 1)

        # -------- Deepest level: InfoNCE --------
        f_t1  = feats_t1[-1]
        f_t1c = feats_t1c[-1]
        f_t2  = feats_t2[-1]

        # Global average pooling
        f_t1  = f_t1.mean(dim=[2, 3])  # [B,C,H,W] -> [B,C]
        f_t1c = f_t1c.mean(dim=[2, 3]) # [B,C,H,W] -> [B,C]
        f_t2  = f_t2.mean(dim=[2, 3])  # [B,C,H,W] -> [B,C]

        # [3B, D] with slice-consistent ordering
        feats_nce = torch.stack(
            [f_t1, f_t1c, f_t2], dim=1
        ).view(-1, f_t1.shape[1])
        # stack: [B, 3, C] -> view: [3B, C]
        # eg. B=2, C=4:
        # f_t1 = [[a1,a2,a3,a4],       f_t1c = [[b1,b2,b3,b4],       f_t2 = [[c1,c2,c3,c4],
        #         [a5,a6,a7,a8]]                [b5,b6,b7,b8]]               [c5,c6,c7,c8]]

        # stack -> [[[a1,a2,a3,a4], [b1,b2,b3,b4], [c1,c2,c3,c4]],
        #           [[a5,a6,a7,a8], [b5,b6,b7,b8], [c5,c6,c7,c8]]]

        # view  -> [[a1,a2,a3,a4],
        #           [b1,b2,b3,b4],
        #           [c1,c2,c3,c4],
        #           [a5,a6,a7,a8],
        #           [b5,b6,b7,b8],
        #           [c5,c6,c7,c8]]

        nce_loss = self.nce_loss(feats_nce) # input [3B, D]

        total_loss = (
            self.mae_weight * mae_loss +
            self.nce_weight * nce_loss
        )

        return {
            "total": total_loss,
            "mae": mae_loss,
            "nce": nce_loss,
        }


class CLIPLoss(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, images, txt_feat):
        # images: [B, 1, H, W] -> repeat to 3 channel
        images = images.repeat(1, 3, 1, 1)
        img_feat = self.clip.get_image_features(images) # [B, D]
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return 1 - (img_feat * txt_feat).sum(dim=-1).mean()