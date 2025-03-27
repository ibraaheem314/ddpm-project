# train.py (Training DDPM on CIFAR-10 in PyTorch)

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
from model import UNet
from diffusion import Diffusion
from tqdm import tqdm

def main():
    # Configurations
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', DEVICE)
    if DEVICE == 'cuda':
        print('GPU Name:', torch.cuda.get_device_name(0))
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 2e-4

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Model and Diffusion
    model = UNet().to(DEVICE)
    diffusion = Diffusion(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    global_step = 0

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader)
        for images, _ in pbar:
            images = images.to(DEVICE)
            t = torch.randint(0, diffusion.timesteps, (images.size(0),), device=DEVICE).long()

            loss = diffusion.loss(model, images, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            pbar.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}")

        # Save checkpoint at the end of the last epoch
        if epoch + 1 == EPOCHS:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()