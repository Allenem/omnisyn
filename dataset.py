import os
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, train_dir, transform=None):
        """
        Args:
            train_dir (str): train 路径
            transform: torchvision transform（同时用于 input / target）
        """
        self.train_dir = train_dir
        self.transform = transform

        self.samples = []  # [(t1_path, t1c_path, t2_path), ...]

        cases = sorted(os.listdir(train_dir))

        for case in cases:
            case_dir = os.path.join(train_dir, case)
            if not os.path.isdir(case_dir):
                continue

            t1_dir = os.path.join(case_dir, "T1")
            t1c_dir = os.path.join(case_dir, "T1c_reg")
            t2_dir = os.path.join(case_dir, "T2_reg")

            t1_files = sorted(os.listdir(t1_dir))
            t1c_files = sorted(os.listdir(t1c_dir))
            t2_files = sorted(os.listdir(t2_dir))

            assert len(t1_files) == len(t1c_files) == len(t2_files), \
                f"Slice number mismatch in {case}"

            for f1, f1c, f2 in zip(t1_files, t1c_files, t2_files):
                self.samples.append((
                    os.path.join(t1_dir, f1),
                    os.path.join(t1c_dir, f1c),
                    os.path.join(t2_dir, f2),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t1_path, t1c_path, t2_path = self.samples[idx]

        t1 = Image.open(t1_path).convert("L")
        t1c = Image.open(t1c_path).convert("L")
        t2 = Image.open(t2_path).convert("L")

        if self.transform is not None:
            t1 = self.transform(t1)
            t1c = self.transform(t1c)
            t2 = self.transform(t2)

        return {
            "T1": t1,
            "T1c": t1c,
            "T2": t2,
        }


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = MyDataset(train_dir="H:/251118npcdata/train/", transform=transform)
    print(f"Dataset size: {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataloader size: {len(dataloader)} batches")

    for batch in dataloader:
        t1 = batch["T1"]
        t1c = batch["T1c"]
        t2 = batch["T2"]
        print(t1.shape, t1c.shape, t2.shape)
        # 画出来
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(t1[0][0], cmap='gray')
        axs[0].set_title("T1")
        axs[1].imshow(t1c[0][0], cmap='gray')
        axs[1].set_title("T1c")
        axs[2].imshow(t2[0][0], cmap='gray')
        axs[2].set_title("T2")
        plt.show()
        break

    # Dataset size: 30825 samples
    # Dataloader size: 7707 batches
    # torch.Size([4, 1, 224, 224]) torch.Size([4, 1, 224, 224]) torch.Size([4, 1, 224, 224])