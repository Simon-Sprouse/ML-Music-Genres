import os
from PIL import Image
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm


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