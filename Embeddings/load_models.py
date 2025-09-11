import numpy as np
import os
import torch

import model_classes

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
    
    

def loadSimpleAE(pth_path):
    
    if pth_path is None:
        return None

    if not os.path.isfile(pth_path):
        return None
    
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded_model = model_classes.SimpleAutoencoder(img_size=128, embedding_dim=128).to(device)
        loaded_model.load_state_dict(torch.load(pth_path, map_location=device))
        loaded_model.eval()
        return loaded_model
    except Exception as e:
        print(f"⚠️ Error loading simple autoencoder from {pth_path}: {e}")
        return None

    
def loadConvAE(pth_path):
    
    if pth_path is None:
        return None

    if not os.path.isfile(pth_path):
        return None
    
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded_model = model_classes.ConvAutoencoder(latent_dim=128).to(device)
        loaded_model.load_state_dict(torch.load(pth_path, map_location=device))
        loaded_model.eval()
        return loaded_model
    except Exception as e:
        print(f"⚠️ Error loading conv autoencoder from {pth_path}: {e}")
        return None

    
    
    
  
def loadConvVAE(pth_path):
    
    if pth_path is None:
        return None

    if not os.path.isfile(pth_path):
        return None
    
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded_model = model_classes.ConvVAE(latent_dim=128).to(device)
        loaded_model.load_state_dict(torch.load(pth_path, map_location=device))
        loaded_model.eval()
        return loaded_model
    except Exception as e:
        print(f"⚠️ Error loading conv VAE from {pth_path}: {e}")
        return None
