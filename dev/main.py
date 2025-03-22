# dev/main.py
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_sklearn


# Generator Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels and out_channels differ, use a 1x1 conv for shortcut
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        out = self.relu(out)

        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.init_size = 4  # Initial image size (4x4)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),  # 8x8
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),  # 16x16
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),  # 32x32
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )


    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Critic Model (Discriminator)
class Critic(nn.Module):
    def __init__(self, img_channels):
        super(Critic, self).__init__()
        def critic_block(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.model = nn.Sequential(
            *critic_block(img_channels, 64),
            *critic_block(64, 128),
            *critic_block(128, 256),

            *critic_block(256, 512),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # Output Wasserstein distance
        )

    def forward(self, img):
        return self.model(img)

# Gradient Penalty Function
def compute_gradient_penalty(critic, real_imgs, fake_imgs, device):
    """Compute the gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_imgs.size(0), 1).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# WGAN Training
def train_wgan(generator, critic, dataloader, latent_dim, epochs, sample_interval=1, lambda_gp=10):
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_C = optim.Adam(critic.parameters(), lr=0.00005, betas=(0.5, 0.999))

    g_losses, c_losses, ssim_scores = [], [], []

    for epoch in range(epochs):
        g_loss_epoch, c_loss_epoch = 0.0, 0.0
        epoch_ssim_scores = []

        for i, imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            # Train Critic
            optimizer_C.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            real_loss = torch.mean(critic(real_imgs))
            fake_loss = torch.mean(critic(fake_imgs))
            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)
            c_loss = fake_loss - real_loss + lambda_gp * gradient_penalty
            c_loss.backward()
            optimizer_C.step()
            c_loss_epoch += c_loss.item()

            # Train Generator every n_critic steps
            if i % 8 == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, latent_dim).to(device)
                gen_imgs = generator(z)
                g_loss = -torch.mean(critic(gen_imgs))
                g_loss.backward()
                optimizer_G.step()
                g_loss_epoch += g_loss.item()

                # Calculate SSIM periodically
                ssim_score = calculate_ssim(real_imgs, gen_imgs)
                epoch_ssim_scores.append(ssim_score)

        # Record losses and SSIM
        g_losses.append(g_loss_epoch / len(dataloader))
        c_losses.append(c_loss_epoch / len(dataloader))
        ssim_scores.append(np.mean(epoch_ssim_scores))

        # Save sample images
        if (epoch + 1) % sample_interval == 0:
            display_sample_images(generator, latent_dim)

        print(f"[Epoch {epoch + 1}] Generator Loss: {g_losses[-1]:.4f}, Critic Loss: {c_losses[-1]:.4f}, SSIM: {ssim_scores[-1]:.4f}")

    # Plot losses and SSIM
    plot_losses_and_ssim(g_losses, c_losses, ssim_scores)

# Display and Plot Functions
def display_sample_images(generator, latent_dim):
    generator.eval()
    z = torch.randn(16, latent_dim).to(device)
    gen_imgs = generator(z).detach().cpu()
    gen_imgs = (gen_imgs + 1) / 2  # Rescale to [0, 1]
    grid = make_grid(gen_imgs, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    generator.train()

def plot_losses_and_ssim(g_losses, c_losses, ssim_scores):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Loss plot
    ax1.plot(g_losses, label="Generator Loss")
    ax1.plot(c_losses, label="Critic Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # SSIM plot
    ax2.plot(ssim_scores, label="SSIM Score", color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.legend()

    plt.tight_layout()
    plt.show()

def calculate_ssim(real_imgs, gen_imgs, win_size=3):
    """
    Calculate SSIM between real and generated images

    Args:
    real_imgs (torch.Tensor): Real image batch
    gen_imgs (torch.Tensor): Generated image batch
    win_size (int): Window size for SSIM calculation (should be smaller than the image size)

    Returns:
    float: Average SSIM score
    """
    # Convert images to numpy for SSIM calculation
    real_np = real_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gen_np = gen_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)

    # Normalize images to [0, 1] range
    real_np = (real_np + 1) / 2
    gen_np = (gen_np + 1) / 2

    # Compute SSIM for each image with custom window size and data_range specified
    ssim_scores = [ssim_sklearn(real_np[i], gen_np[i], multichannel=True, win_size=win_size, data_range=1.0)
                   for i in range(len(real_np))]

    return np.mean(ssim_scores)

# Hyperparameters and Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
img_channels = 3
img_size = 64
epochs = 300
batch_size = 64
lambda_gp = 10

generator = Generator(latent_dim, img_channels).to(device)
critic = Critic(img_channels).to(device)

data_path = "/home/kaggle/data/mcskin/1/skins"  # Replace with your dataset directory
dataloader = load_minecraft_data(data_path, img_size=img_size, batch_size=batch_size)

    # Initialize and Train WGAN
train_wgan(generator, critic, dataloader, latent_dim, epochs, sample_interval=5, lambda_gp=lambda_gp)
torch.save(generator.state_dict(), f"generator_epoch_{epochs}.pth")
torch.save(critic.state_dict(), f"critic_epoch_{epochs}.pth")
