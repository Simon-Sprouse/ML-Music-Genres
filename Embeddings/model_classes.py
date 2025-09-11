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