import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, img_size=128, embedding_dim=128):
        super().__init__()
        # Encoder
        self.img_size = img_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = recon.view(-1, 1, self.img_size, self.img_size)
        return recon, z
    
    
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 1x128x128 -> 16x64x64
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 128x8x8
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 64x16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 32x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # 16x64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # 1x128x128
            nn.Sigmoid(),  # keep values in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        z = self.fc_enc(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 128, 8, 8)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    
    
    
    
    
    
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 1x128x128 -> 16x64x64
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 128x8x8
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()
        hidden_dim = 128 * 8 * 8
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 64x16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 32x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # 16x64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # 1x128x128
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 128, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon_loss + beta * kl_loss) / x.size(0)

