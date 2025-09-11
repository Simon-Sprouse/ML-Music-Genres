import os
import glob
from PIL import Image
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import model_classes



def build_memmap(image_paths, memmap_path, img_size=(128, 128), dtype=np.float32):
    """
    Create a memmap of shape (N, D) where D = H*W, grayscale and flattened.
    """
    N = len(image_paths)
    H, W = img_size
    D = H * W
    os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
    mm = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=(N, D))
    for i, p in enumerate(tqdm(image_paths, desc="Writing memmap")):
        img = Image.open(p).convert('L').resize((W, H), Image.BILINEAR)
        arr = np.asarray(img, dtype=dtype).reshape(-1) / 255.0
        mm[i, :] = arr
    mm.flush()
    return memmap_path, (N, D)

def run_incremental_pca(memmap_path, shape, n_components=100, batch_size=64, save_path="pixel_pca.npz"):
    N, D = shape
    mm = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=(N, D))

    ipca = IncrementalPCA(n_components=n_components)
    # First pass: fit in batches to compute principal components
    for start in tqdm(range(0, N, batch_size), desc="IPCA fit"):
        batch = mm[start:start+batch_size]
        ipca.partial_fit(batch)
    # Optionally transform all data (second pass)
    components = ipca.components_          # shape (n_components, D)
    mean = ipca.mean_                      # shape (D,)
    explained = ipca.explained_variance_ratio_

    np.savez_compressed(save_path,
                        components=components,
                        mean=mean,
                        explained=explained,
                        n_components=n_components)
    print(f"Saved PCA data to {save_path}")



def fitPCA(filenames, model_save_dir):

    memmap_path = f"{model_save_dir}/pixel_memmap.dat"
    save_path=f"{model_save_dir}/pixel_pca.npz"
    memmap_path, shape = build_memmap(filenames, memmap_path, img_size=(128,128))
    run_incremental_pca(memmap_path, shape, n_components=50, batch_size=64, save_path=save_path)
    
    
    
    
    
    
    
    
    
    
    
    
class AE_Dataset(Dataset):
    def __init__(self, filenames=None, root_dir=None, img_size=128):
        """
        Args:
            filenames (list[str]): Explicit list of image paths to load.
            root_dir (str): Directory containing song folders with spectrograms.
                            If provided, will auto-discover aligned_spectrogram images.
            img_size (int): Final resize to (img_size, img_size).
        """
        if filenames is not None:
            self.files = filenames
        elif root_dir is not None:
            self.files = []
            for song_dir in glob(os.path.join(root_dir, "*")):
                self.files.extend(
                    [f for f in glob(os.path.join(song_dir, "*.png")) if "aligned_spectrogram" in f]
                )
        else:
            raise ValueError("Must provide either filenames or root_dir")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # scales to [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")  # grayscale
        return self.transform(img)
    
    
    
def trainSimpleAE(train_dataloader, test_dataloader, model_save_dir):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_classes.SimpleAutoencoder(img_size=128, embedding_dim=128).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Simple Autoencoder")
    print("Model ready on", device)
    
    
    n_epochs = 2
    for epoch in range(n_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for imgs in tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{n_epochs}"):
            imgs = imgs.to(device)
            recon, _ = model(imgs)  # your model returns (recon, z)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_dataloader.dataset)

        # --- Evaluation phase ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs in tqdm(test_dataloader, desc=f"Test Epoch {epoch+1}/{n_epochs}"):
                imgs = imgs.to(device)
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                test_loss += loss.item() * imgs.size(0)

        test_loss = test_loss / len(test_dataloader.dataset)

        # --- Logging ---
        print(f"Epoch [{epoch+1}/{n_epochs}] "
            f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
    save_path = os.path.join(model_save_dir, "autoencoder.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    print("saved autoencoder at: ", save_path)
    
    
    
    
def trainConvAE(train_dataloader, test_dataloader, model_save_dir):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model_classes.ConvAutoencoder(latent_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    print("Training Conv Autoencoder")
    print("Model ready on", device)
    
    
    n_epochs = 2
    for epoch in range(n_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for imgs in tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{n_epochs}"):
            imgs = imgs.to(device)
            recon, _ = model(imgs)  # your model returns (recon, z)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_dataloader.dataset)

        # --- Evaluation phase ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs in tqdm(test_dataloader, desc=f"Test Epoch {epoch+1}/{n_epochs}"):
                imgs = imgs.to(device)
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                test_loss += loss.item() * imgs.size(0)

        test_loss = test_loss / len(test_dataloader.dataset)

        # --- Logging ---
        print(f"Epoch [{epoch+1}/{n_epochs}] "
            f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
    save_path = os.path.join(model_save_dir, "conv_autoencoder.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    print("saved conv autoencoder at: ", save_path)
    
    
    
    
def trainConvVAE(train_dataloader, test_dataloader, model_save_dir, beta=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_classes.ConvVAE(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Conv VAE")
    print("Model ready on", device)

    n_epochs = 2
    for epoch in range(n_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for imgs in tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{n_epochs}"):
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = model_classes.vae_loss(recon, imgs, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_dataloader.dataset)

        # --- Evaluation phase ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs in tqdm(test_dataloader, desc=f"Test Epoch {epoch+1}/{n_epochs}"):
                imgs = imgs.to(device)
                recon, mu, logvar = model(imgs)
                loss = model_classes.vae_loss(recon, imgs, mu, logvar, beta=beta)
                test_loss += loss.item() * imgs.size(0)

        test_loss = test_loss / len(test_dataloader.dataset)

        # --- Logging ---
        print(f"Epoch [{epoch+1}/{n_epochs}] "
              f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    save_path = os.path.join(model_save_dir, "conv_variational_autoencoder.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print("saved conv VAE at: ", save_path)
