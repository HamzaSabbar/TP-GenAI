import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

BATCH_SIZE = 128
K = 32
EPOCHS = 10
LR = 1e-3


class MNISTFlattenDataset(Dataset):
    def __init__(self, train=True):
        self.mnist = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        image = image.view(-1)
        return image, label


class Autoencoder(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, k),
        )

        self.decoder = nn.Sequential(
            nn.Linear(k, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


def train_autoencoder(model, train_loader, criterion, optimizer, device):
    print("\nDebut de l'entrainement...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

    print("\nEntrainement termine.\n")


def collect_loader_arrays(loader):
    images_list = []
    labels_list = []

    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)

    images_array = torch.cat(images_list, dim=0).numpy()
    labels_array = torch.cat(labels_list, dim=0).numpy()
    return images_array, labels_array


def reconstruct_with_autoencoder(model, test_loader, device):
    model.eval()
    ae_reconstructions = []
    ae_latent_vectors = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            z = model.encoder(images)
            reconstructed = model.decoder(z)

            ae_latent_vectors.append(z.cpu())
            ae_reconstructions.append(reconstructed.cpu())

    ae_latent_vectors = torch.cat(ae_latent_vectors, dim=0).numpy()
    ae_reconstructions = torch.cat(ae_reconstructions, dim=0).numpy()
    return ae_latent_vectors, ae_reconstructions


def plot_reconstructions(x_test, ae_reconstructions, x_test_pca_reconstructed, n=5):
    plt.figure(figsize=(12, 6))

    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(3, n, n + i + 1)
        plt.imshow(ae_reconstructions[i].reshape(28, 28), cmap="gray")
        plt.title("AE")
        plt.axis("off")

        plt.subplot(3, n, (2 * n) + i + 1)
        plt.imshow(x_test_pca_reconstructed[i].reshape(28, 28), cmap="gray")
        plt.title("PCA")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_tsne(ae_latent_vectors, x_test_pca, y_test, sample_size=2000):
    print("\nCalcul de t-SNE...\n")

    x_ae_sample = ae_latent_vectors[:sample_size]
    x_pca_sample = x_test_pca[:sample_size]
    y_sample = y_test[:sample_size]

    tsne_ae = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    )
    ae_2d = tsne_ae.fit_transform(x_ae_sample)

    tsne_pca = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    )
    pca_2d = tsne_pca.fit_transform(x_pca_sample)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(ae_2d[:, 0], ae_2d[:, 1], c=y_sample, cmap="tab10", s=8)
    plt.title("t-SNE - Autoencoder")
    plt.subplot(1, 2, 2)
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_sample, cmap="tab10", s=8)
    plt.title("t-SNE - PCA")
    plt.tight_layout()
    plt.show()


def print_observations():
    print("\nConclusion / observations :")
    print("- L'autoencoder apprend une representation non lineaire des chiffres.")
    print("- PCA fournit une reduction lineaire plus simple et rapide.")
    print("- Les reconstructions de l'autoencoder sont souvent plus fideles que celles de PCA.")
    print("- t-SNE permet de visualiser la separation des classes dans l'espace reduit.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilise :", device)

    train_dataset = MNISTFlattenDataset(train=True)
    test_dataset = MNISTFlattenDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Taille train :", len(train_dataset))
    print("Taille test  :", len(test_dataset))

    model = Autoencoder(K).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_autoencoder(model, train_loader, criterion, optimizer, device)

    x_train, _ = collect_loader_arrays(train_loader)
    x_test, y_test = collect_loader_arrays(test_loader)

    print("Shape X_train:", x_train.shape)
    print("Shape X_test :", x_test.shape)

    ae_latent_vectors, ae_reconstructions = reconstruct_with_autoencoder(
        model, test_loader, device
    )
    print("Shape latent autoencoder:", ae_latent_vectors.shape)

    print("\nApplication de PCA...\n")

    pca = PCA(n_components=K)
    pca.fit(x_train)

    x_test_pca = pca.transform(x_test)
    x_test_pca_reconstructed = pca.inverse_transform(x_test_pca)

    print("Shape PCA reduit:", x_test_pca.shape)

    plot_reconstructions(x_test, ae_reconstructions, x_test_pca_reconstructed, n=5)

    plot_tsne(ae_latent_vectors, x_test_pca, y_test, sample_size=2000)

    print_observations()
    print("\nTP termine avec succes.")


if __name__ == "__main__":
    main()
