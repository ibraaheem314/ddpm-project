import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import UNet
import torch.nn.functional as F
from diffusion import Diffusion
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config("config.yml")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = config["training"]["batch_size"]
    LR = config["training"]["learning_rate"]
    EPOCHS = config["training"]["epochs"]
    T = config["diffusion"]["timesteps"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(**config["model"]).to(DEVICE)
    diffusion = Diffusion(T=T, beta_schedule=config["diffusion"]["beta_schedule"])

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x0, _ in tqdm(train_loader):
            x0 = x0.to(DEVICE)
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, torch.cat([noise, torch.zeros_like(noise)], dim=1))  # Adaptez selon votre sortie
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()