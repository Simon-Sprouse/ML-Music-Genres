import numpy as np
import torch
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