# train.py (Training DDPM on CIFAR-10 in PyTorch)

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
import copy
from model import UNet
from diffusion import Diffusion
from tqdm import tqdm

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

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
    ema_model = copy.deepcopy(model)
    ema_model.eval()
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

            update_ema(ema_model, model)

            global_step += 1
            pbar.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}")

        # Save checkpoint at the end of the last epoch
        if epoch + 1 == EPOCHS:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
            torch.save(ema_model.state_dict(), f'model_ema_epoch_{epoch+1}.pth')
            
        # Comparaison finale entre le modèle entraîné et l'EMA
        
    total_diff = 0.
    total_norm = 0.
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        diff = (param.data - ema_param.data).abs().mean().item()
        total_diff += diff
        total_norm += param.data.abs().mean().item()

    print(f"\n📊 Moyenne des écarts absolus entre model et ema_model : {total_diff:.6f}")
    print(f"📊 Norme moyenne des poids du model : {total_norm:.6f}")
    print(f"📉 Ratio de différence relative : {(total_diff / total_norm):.4%}")


if __name__ == '__main__':
    main()
