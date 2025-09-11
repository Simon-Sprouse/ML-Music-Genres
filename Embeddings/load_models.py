import numpy as np
import os

def loadPCA(npz_path):
    """
    Reload trained PCA from .npz file.
    Returns (components, mean, n_components) if successful,
    or None if loading fails.
    """
    if npz_path is None:
        ## print("⚠️ No PCA file path provided.")
        return None

    if not os.path.isfile(npz_path):
        ## print(f"⚠️ PCA file not found: {npz_path}")
        return None

    try:
        data = np.load(npz_path)
        components = data["components"]      # (n_components, D)
        mean = data["mean"]                  # (D,)
        n_components = int(data["n_components"])
        return components, mean, n_components
    except Exception as e:
        print(f"⚠️ Error loading PCA from {npz_path}: {e}")
        return None