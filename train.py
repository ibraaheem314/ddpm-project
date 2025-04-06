import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
from diffusion import Diffusion
from model import UNet

def train():
    # Configuration de l'appareil (GPU si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configurations
    batch_size = 64  # Adapté pour 8 Go de VRAM
    lr = 2e-4
    epochs = 100
    image_size = 32
    
    # Modèle
    model = UNet(
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2
    ).to(device)
    
    # Diffusion
    diffusion = Diffusion(T=1000, beta_schedule='cosine', device=device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Préparation du dataset CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
            
            optimizer.zero_grad()
            loss = diffusion.p_losses(model, x, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        # Sauvegarde du modèle tous les 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    train()
