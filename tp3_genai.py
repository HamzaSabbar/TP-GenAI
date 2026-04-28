import kagglehub
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt


path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

ld = 100
batch_taille = 128
n = 5
lr = 0.0002
taille_image = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


transform = transforms.Compose([
    transforms.Resize(taille_image),
    transforms.CenterCrop(taille_image),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


DATA_PATH = os.path.join(path, "img_align_celeba", "img_align_celeba")


class CelebAFlat(Dataset):
    def __init__(self, folder, transform=None):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, 0


dataset = CelebAFlat(DATA_PATH, transform=transform)

dataloader = DataLoader(
    dataset,
    batch_size=batch_taille,
    shuffle=True,
    num_workers=2,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=True
)

print(f"{len(dataset)} images chargées")


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(ld, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)


criterion = nn.BCELoss()

opt_G = torch.optim.Adam(
    G.parameters(),
    lr=lr,
    betas=(0.5, 0.999)
)

opt_D = torch.optim.Adam(
    D.parameters(),
    lr=lr,
    betas=(0.5, 0.999)
)


fixed_noise = torch.randn(64, ld, 1, 1, device=DEVICE)


for j in range(n):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(DEVICE)

        b = real_imgs.size(0)

        real_labels = torch.ones(b, device=DEVICE)
        fake_labels = torch.zeros(b, device=DEVICE)

        opt_D.zero_grad()

        real_loss = criterion(D(real_imgs), real_labels)

        noise = torch.randn(b, ld, 1, 1, device=DEVICE)
        fake_imgs = G(noise)

        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)

        loss_D = real_loss + fake_loss

        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()

        noise = torch.randn(b, ld, 1, 1, device=DEVICE)
        fake_imgs = G(noise)

        loss_G = criterion(D(fake_imgs), real_labels)

        loss_G.backward()
        opt_G.step()

        if i % 100 == 0:
            print(
                f"Epoch [{j + 1}/{n}] "
                f"Step [{i}/{len(dataloader)}] "
                f"D_loss: {loss_D.item():.4f} "
                f"G_loss: {loss_G.item():.4f}"
            )


G.eval()

with torch.no_grad():
    fake = G(fixed_noise).cpu()


grid = vutils.make_grid(
    fake,
    nrow=8,
    normalize=True
).permute(1, 2, 0).numpy()


plt.figure(figsize=(12, 12))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(grid)
plt.show()