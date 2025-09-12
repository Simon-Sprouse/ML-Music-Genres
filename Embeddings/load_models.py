import numpy as np
import os
import torch
import torchvision.models
import timm

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







# Utility: freeze params
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_Cnn_Models():
    

    # 1. ResNet-34
    resnet34 = freeze_model(torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1))
    print("ResNet-34 loaded:", sum(p.numel() for p in resnet34.parameters())/1e6, "M params")

    # 2. InceptionV3
    inception = freeze_model(torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1))
    print("InceptionV3 loaded:", sum(p.numel() for p in inception.parameters())/1e6, "M params")

    # 3. SqueezeNet 1.1
    squeezenet = freeze_model(torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1))
    print("SqueezeNet loaded:", sum(p.numel() for p in squeezenet.parameters())/1e6, "M params")

    # 4. EfficientNetV2-S (via timm)
    efficientnetv2s = freeze_model(timm.create_model("tf_efficientnetv2_s_in21k", pretrained=True))
    print("EfficientNetV2-S loaded:", sum(p.numel() for p in efficientnetv2s.parameters())/1e6, "M params")

    # 5. MobileNetV3-Small
    mobilenetv3s = freeze_model(torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1))
    print("MobileNetV3-Small loaded:", sum(p.numel() for p in mobilenetv3s.parameters())/1e6, "M params")

    print("\n✅ All models loaded and frozen successfully!")
    
    models =  {
        "resnet34": resnet34,
        "inceptionv3": inception,
        "squeezenet1.1": squeezenet,
        "efficientnetv2-s": efficientnetv2s,
        "mobilenetv3s": mobilenetv3s
    }


    return models