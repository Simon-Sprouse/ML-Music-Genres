import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm


def runPCA(filenames, components, mean, n_components, img_size=(128,128)):
    H, W = img_size

    all_embeds = []

    for fn in tqdm(filenames, desc="Projecting images (PCA)"):
        # Load + preprocess image
        img = Image.open(fn).convert("L").resize((W, H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

        # Center and project
        arr_centered = arr - mean
        embed = np.dot(components, arr_centered)   # shape (n_components,)

        all_embeds.append(embed)

    # Stack into (N, n_components)
    embeddings_tensor = torch.tensor(np.stack(all_embeds), dtype=torch.float32)

    return embeddings_tensor



def runAE(filenames, model, device=None):
    """
    Compute embeddings for a list of image filenames using a trained autoencoder.
    
    Args:
        model: Trained SimpleAutoencoder (already on device, eval mode).
        filenames: list of image file paths.
        batch_size: number of images per forward pass.
        device: 'cuda' or 'cpu'. If None, inferred from model.
    
    Returns:
        embeddings: torch.Tensor of shape (len(filenames), embedding_dim)
    """
    if device is None:
        device = next(model.parameters()).device

    # --- Preprocessing ---
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # force 1 channel
        transforms.ToTensor(),  # [0,1], shape (1, H, W)
    ])

    # --- Load images ---
    imgs = []
    for fname in filenames:
        img = Image.open(fname).convert("L")  # grayscale
        img = transform(img)  # (1, H, W)
        imgs.append(img)

    imgs = torch.stack(imgs)  # (N, 1, H, W)

    # --- Generate embeddings in batches ---
    embeddings = []
    model.eval()
    with torch.no_grad():

        imgs = imgs.to(device)
        _, z = model(imgs)
        embeddings.append(z.cpu())
    
    return torch.cat(embeddings, dim=0)  # (N, embedding_dim)
