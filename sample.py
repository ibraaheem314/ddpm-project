import torch
from model import UNet
from diffusion import Diffusion
import matplotlib.pyplot as plt

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Charger le modèle et son checkpoint
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch_100.pt", map_location=device))
    model.eval()
    
    # Initialiser diffusion
    diffusion = Diffusion(T=1000, beta_schedule='cosine', device=device)
    
    # Génération d'échantillons
    samples = diffusion.sample(model, (16, 3, 32, 32), device)
    
    # Sauvegarder le tenseur et une image visuelle
    torch.save(samples, "generated/samples.pt")
    
    samples = (samples + 1) / 2  # Dés-normalisation pour l'affichage (dans [0,1])
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i].cpu().permute(1, 2, 0))
        plt.axis("off")
    plt.savefig("generated/samples.png")
    plt.show()

if __name__ == "__main__":
    main()
