import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
from diffusion import*
from model import*


def train():
    # Config
    batch_size = 64  # Adapté pour 8 Go de VRAM
    lr = 2e-4
    epochs = 20
    image_size = 32
    
    # Modèle
    model = UNet(
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2
    ).cuda()
    
    # Diffusion
    diffusion = Diffusion(T=1000, beta_schedule='cosine', device='cuda')
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Entraînement
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, _ in tqdm(loader):
            x = x.cuda()
            t = torch.randint(0, diffusion.T, (x.size(0),), device='cuda')
            
            optimizer.zero_grad()
            loss = diffusion.p_losses(model, x, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    train()