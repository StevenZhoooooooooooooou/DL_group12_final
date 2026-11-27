"""VAE architectures used in the four experiments.

Every model exposes encode(x) -> (mu, logvar), decode(z), forward(x) -> (x_hat, mu, logvar)
and `attn_layers`: a dict name -> module whose output can be used as the feature map A.
"""
import torch
import torch.nn as nn
import torchvision


def reparameterize(mu, logvar):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class _VAE(nn.Module):
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar


class MNISTVAE(_VAE):
    """Architecture of the official expVAE code (liuem607/expVAE, code/model.py)."""

    def __init__(self, z_dim=32):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU())     # 14x14
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU())   # 7x7  (official target 'encoder.2')
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 7 * 7, 1024), nn.ReLU())
        self.fc_mu, self.fc_lv = nn.Linear(1024, z_dim), nn.Linear(1024, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 1024), nn.ReLU(), nn.Linear(1024, 128 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid())
        self.attn_layers = {"conv1": self.conv1, "conv2": self.conv2}

    def encode(self, x):
        h = self.fc(self.conv2(self.conv1(x)))
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(z)


class Ped1VAE(_VAE):
    """100x100 grey-scale VAE with three stride-2 convs: 50x50, 25x25, 12x12 (paper Table 1)."""

    MEAN, STD = 0.3750, 0.2013

    def __init__(self, z_dim=32):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())      # 50
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())   # 25
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())  # 12
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 12 * 12, 1024), nn.ReLU())
        self.fc_mu, self.fc_lv = nn.Linear(1024, z_dim), nn.Linear(1024, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 1024), nn.ReLU(), nn.Linear(1024, 256 * 12 * 12), nn.ReLU(),
            nn.Unflatten(1, (256, 12, 12)),
            nn.ConvTranspose2d(256, 128, 5, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),   # 25
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),     # 50
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid())                       # 100
        self.attn_layers = {"conv1": self.conv1, "conv2": self.conv2, "conv3": self.conv3}

    def encode(self, x):
        h = self.fc(self.conv3(self.conv2(self.conv1((x - self.MEAN) / self.STD))))
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(z)


class ResNet18VAE(_VAE):
    """MVTec-AD: ResNet18 feature encoder + 32-d latent (paper Sec. 3.3.1), 256x256 RGB.

    Feature maps at 256x256 input: layer1 64x64, layer2 32x32, layer3 16x16, layer4 8x8.
    """

    def __init__(self, z_dim=32, pretrained=False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.fc_mu, self.fc_lv = nn.Linear(512, z_dim), nn.Linear(512, z_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def up(cin, cout):
            return [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 512 * 4 * 4), nn.ReLU(True), nn.Unflatten(1, (512, 4, 4)),
            *up(512, 256), *up(256, 256), *up(256, 128), *up(128, 64), *up(64, 32), *up(32, 16),  # 4 -> 256
            nn.Conv2d(16, 3, 3, 1, 1), nn.Sigmoid())
        self.attn_layers = {"layer1": self.layer1, "layer2": self.layer2,
                            "layer3": self.layer3, "layer4": self.layer4}

    def encode(self, x):
        h = self.stem((x - self.mean) / self.std)
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        h = h.mean(dim=(2, 3))
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(z)


class FactorVAENet(_VAE):
    """dSprites encoder/decoder of Kim & Mnih (2018), as in 1Konny/FactorVAE. Decoder returns logits."""

    def __init__(self, z_dim=10, init_deconv=True):
        super().__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True))    # 32
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True))   # 16
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True))   # 8
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True))   # 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(True), nn.Linear(128, 2 * z_dim))
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(True), nn.Linear(128, 1024), nn.ReLU(True),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))
        self.attn_layers = {"conv1": self.conv1, "conv2": self.conv2, "conv3": self.conv3}
        # 1Konny/FactorVAE applies N(0, 0.02) to Linear/Conv2d only; init_deconv=False reproduces that
        kinds = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d) if init_deconv else (nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, kinds):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        s = self.head(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return s[:, :self.z_dim], s[:, self.z_dim:]

    def decode(self, z):
        return self.dec(z)


class Discriminator(nn.Module):
    """FactorVAE total-correlation discriminator (6-layer MLP, 1000 units)."""

    def __init__(self, z_dim=10):
        super().__init__()
        layers, d = [], z_dim
        for _ in range(5):
            layers += [nn.Linear(d, 1000), nn.LeakyReLU(0.2, True)]
            d = 1000
        self.net = nn.Sequential(*layers, nn.Linear(1000, 2))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------------------------
# Architectures exactly as listed in the paper's supplementary material (Tables 2-4).
# ---------------------------------------------------------------------------------------------

class Ped1VAESupp(_VAE):
    """Supplement Table 2: three stride-2 convs (64/128/256) with ReLU and no BatchNorm, 36864 -> 1024 -> 32."""

    MEAN, STD = 0.3750, 0.2013

    def __init__(self, z_dim=32):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU())      # 50
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU())    # 25
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())   # 12
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(36864, 1024), nn.ReLU())
        self.fc_mu, self.fc_lv = nn.Linear(1024, z_dim), nn.Linear(1024, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 1024), nn.ReLU(), nn.Linear(1024, 36864), nn.ReLU(),
            nn.Unflatten(1, (256, 12, 12)), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 1), nn.ReLU(),                   # 25
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),                    # 50
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid())                   # 100
        self.attn_layers = {"conv1": self.conv1, "conv2": self.conv2, "conv3": self.conv3}

    def encode(self, x):
        h = self.fc(self.conv3(self.conv2(self.conv1((x - self.MEAN) / self.STD))))
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(z)


class ResNet18VAESupp(_VAE):
    """Supplement Table 3: ResNet18 without its last 2 layers (8x8x512 at 256x256), *flattened* -> 1024 -> 32,
    decoder 32 -> 1024 -> 1024x4x4 -> 6 x (ConvTranspose + BN + ReLU) -> 256x256x3. The table lists no
    non-linearity between the linear layers, so none is used."""

    def __init__(self, z_dim=32, pretrained=False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512 * 8 * 8, 1024))
        self.fc_mu, self.fc_lv = nn.Linear(1024, z_dim), nn.Linear(1024, z_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def up(cin, cout):
            return [nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(True)]
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 1024), nn.Linear(1024, 1024 * 4 * 4), nn.Unflatten(1, (1024, 4, 4)),
            *up(1024, 512), *up(512, 256), *up(256, 128), *up(128, 64), *up(64, 32),     # 4 -> 128
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())                           # 256
        self.attn_layers = {"layer1": self.layer1, "layer2": self.layer2,
                            "layer3": self.layer3, "layer4": self.layer4}

    def encode(self, x):
        h = self.stem((x - self.mean) / self.std)
        h = self.fc(self.layer4(self.layer3(self.layer2(self.layer1(h)))))
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(z)


class FactorVAENetSupp(_VAE):
    """Supplement Table 4 (= 1Konny FactorVAE1 with z=32): extra 4x4 conv to 1x1x128, 1x1-conv heads,
    1x1-conv / transposed-conv decoder. Decoder returns logits."""

    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True))    # 32
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True))   # 16
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True))   # 8
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),   # 4
                                   nn.Conv2d(64, 128, 4, 1), nn.ReLU(True))     # 1
        self.head = nn.Conv2d(128, 2 * z_dim, 1)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (z_dim, 1, 1)), nn.Conv2d(z_dim, 128, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4), nn.ReLU(True),                     # 4
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),                # 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),                # 16
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),                # 32
            nn.ConvTranspose2d(32, 1, 4, 2, 1))                                # 64
        self.attn_layers = {"conv1": self.conv1, "conv2": self.conv2, "conv3": self.conv3}
        for m in self.modules():                        # 1Konny normal_init: Linear / Conv2d only
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        s = self.head(self.conv4(self.conv3(self.conv2(self.conv1(x))))).flatten(1)
        return s[:, :self.z_dim], s[:, self.z_dim:]

    def decode(self, z):
        return self.dec(z)
