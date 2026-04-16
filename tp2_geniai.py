
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


LATENT_DIM   = 2
HIDDEN_DIM   = 256
INPUT_DIM    = 784        # 28×28
BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FashionMNISTDataset(Dataset):

    def __init__(self, train: bool = True):
        raw = datasets.FashionMNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor()  
        )
        self.images = raw.data.float() / 255.0         
        self.images = self.images.view(-1, INPUT_DIM)  
        self.labels = raw.targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


train_dataset = FashionMNISTDataset(train=True)
test_dataset  = FashionMNISTDataset(train=False)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size : {len(train_dataset)} | Test size : {len(test_dataset)}")



class VAE(nn.Module):

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),   
        )

    def encode(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        z = μ + ε · σ,  avec ε ~ N(0, I)
        std = exp(0.5 · logvar)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)      
            return mu + eps * std
        else:
            return mu  

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar   = self.encode(x)
        z            = self.reparameterize(mu, logvar)
        x_recon      = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    """
    Loss = Reconstruction Loss + KL Divergence

    - Reconstruction : Binary Cross Entropy pixel-wise (somme sur la dim image)
    - KL             : -0.5 · Σ (1 + log σ² - μ² - σ²)
    """
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + kl_loss, recon_loss, kl_loss


model     = VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {"train_loss": [], "recon_loss": [], "kl_loss": []}

print(f"\nEntraînement sur {DEVICE} pendant {EPOCHS} époques...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = recon_total = kl_total = 0.0

    for x, _ in train_loader:
        x = x.to(DEVICE)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss, recon, kl     = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        recon_total += recon.item()
        kl_total    += kl.item()

    n = len(train_loader)
    history["train_loss"].append(total_loss  / n)
    history["recon_loss"].append(recon_total / n)
    history["kl_loss"].append(kl_total    / n)

    print(f"Époque {epoch:2d}/{EPOCHS} | "
          f"Loss: {total_loss/n:.2f} | "
          f"Recon: {recon_total/n:.2f} | "
          f"KL: {kl_total/n:.2f}")

plt.figure(figsize=(10, 4))
plt.plot(history["train_loss"], label="Total Loss")
plt.plot(history["recon_loss"], label="Reconstruction Loss")
plt.plot(history["kl_loss"],    label="KL Divergence")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.title("Évolution de la Loss VAE")
plt.legend()
plt.tight_layout()
plt.savefig("vae_loss.png", dpi=150)
plt.close()
print("vae_loss.png")


#GÉNÉRATION D'IMAGES
model.eval()

with torch.no_grad():

    z_random  = torch.randn(16, LATENT_DIM).to(DEVICE)
    generated = model.decode(z_random).cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    fig.suptitle("Images générées (z ~ N(0, I))", fontsize=13)
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("vae_generated.png", dpi=150)
    plt.close()
    print("Images générées sauvegardées : vae_generated.png")

    grid_size = 15
    z1 = np.linspace(-3, 3, grid_size)
    z2 = np.linspace(-3, 3, grid_size)
    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    for i, zi in enumerate(z1):
        for j, zj in enumerate(z2):
            z      = torch.tensor([[zi, zj]], dtype=torch.float).to(DEVICE)
            img    = model.decode(z).cpu().view(28, 28).numpy()
            canvas[(grid_size - 1 - j) * 28: (grid_size - j) * 28,
                   i * 28: (i + 1) * 28] = img

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.title("Exploration de l'espace latent 2D")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("vae_latent_space.png", dpi=150)
    plt.close()
    print("Espace latent sauvegardé : vae_latent_space.png")

    x_test, _ = next(iter(test_loader))
    x_test     = x_test[:8].to(DEVICE)
    x_recon, _, _ = model(x_test)
    x_test  = x_test.cpu().view(-1, 28, 28)
    x_recon = x_recon.cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    fig.suptitle("Original (haut) vs Reconstruction (bas)", fontsize=13)
    for i in range(8):
        axes[0, i].imshow(x_test[i].numpy(),  cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(x_recon[i].numpy(), cmap="gray"); axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("vae_reconstruction.png", dpi=150)
    plt.close()
    print("vae_reconstruction.png")

