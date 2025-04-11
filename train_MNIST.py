import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from diffusion import Diffusion
from model import UNet
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 512
    lr = 2e-5
    epochs = 1000
    image_size = 28

    # Modèle
    model = UNet(
        in_channels=1,    # MNIST = 1 canal
        out_channels=1,
        model_channels=64,     # Plus de capacité que 64
        channel_mult=(1, 2, 4),    # 2 niveaux
        num_res_blocks=1
    ).to(device)

    diffusion = Diffusion(T=1000, beta_schedule='cosine', device=device)

    # Optimiseur + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5) # Réduit le LR de moitié (gamma=0.5) tous les 20 epochs
    scaler = GradScaler()

    # Chargement Dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = x.to(device)
            # Tirage aléatoire du temps t
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)

            optimizer.zero_grad()

            # En Mixed Precision
            with autocast():
                loss = diffusion.p_losses(model, x, t)

            # Backprop en AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # On remet le grad à zéro (hors loader)
        optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
        scheduler.step()

        # Sauvegarde du checkpoint tous les 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/mnist_epoch_{epoch+1}XXL.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == '__main__':
    train()
