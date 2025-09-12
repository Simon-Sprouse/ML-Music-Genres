import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm


from torch import nn



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



def runVAE(filenames, model, device=None):
    """
    Compute embeddings for a list of image filenames using a trained ConvVAE.
    
    Args:
        model: Trained ConvVAE (already on device, eval mode).
        filenames: list of image file paths.
        device: 'cuda' or 'cpu'. If None, inferred from model.
    
    Returns:
        embeddings: torch.Tensor of shape (len(filenames), latent_dim)
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

    # --- Generate embeddings ---
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        _, mu, logvar = model(imgs)   # forward pass
        embeddings = mu.cpu()         # use mean as embedding
    
    return embeddings  # (N, latent_dim)







def runCNN(filenames, cnn_models):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocessing transforms
    preprocess_224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    preprocess_299 = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    imgs_224 = [preprocess_224(Image.open(f).convert("RGB")) for f in filenames]
    imgs_299 = [preprocess_299(Image.open(f).convert("RGB")) for f in filenames]

    batch_224 = torch.stack(imgs_224).to(device)  # [B, 3, 224, 224]
    batch_299 = torch.stack(imgs_299).to(device)  # [B, 3, 299, 299]

    embeddings = {}

    # Set models to eval
    for model in cnn_models.values():
        model.eval()
    

    
    resnet34 = cnn_models["resnet34"]
    inception = cnn_models["inceptionv3"]
    squeezenet = cnn_models["squeezenet1.1"]
    efficientnet = cnn_models["efficientnetv2-s"]
    mobilenet = cnn_models["mobilenetv3s"]
    

    with torch.no_grad():
        # ResNet-34 - Remove final FC layer
        
        resnet34_feat = torch.flatten(
            torch.nn.Sequential(*list(resnet34.children())[:-1])(batch_224), 1
        )
        embeddings["resnet34"] = resnet34_feat

        # InceptionV3 - Replace FC layer with Identity
        original_fc = inception.fc
        inception.fc = nn.Identity()
        inception_feat = inception(batch_299)
        inception.fc = original_fc
        embeddings["inceptionv3"] = inception_feat

        # SqueezeNet - features + GAP
        sq_raw = squeezenet.features(batch_224)  # [B, 512, H, W]
        sq_pooled = torch.nn.functional.adaptive_avg_pool2d(sq_raw, (1, 1))
        embeddings["squeezenet"] = torch.flatten(sq_pooled, 1)

        # EfficientNetV2-S - forward_features + GAP
        eff_raw = efficientnet.forward_features(batch_224)
        eff_pooled = torch.nn.functional.adaptive_avg_pool2d(eff_raw, (1, 1))
        embeddings["efficientnetv2s"] = torch.flatten(eff_pooled, 1)

        # MobileNetV3-Small - features + GAP
        mob_raw = mobilenet.features(batch_224)
        mob_pooled = torch.nn.functional.adaptive_avg_pool2d(mob_raw, (1, 1))
        embeddings["mobilenetv3_small"] = torch.flatten(mob_pooled, 1)

    return embeddings

