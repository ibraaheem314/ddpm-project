# sample.py
import torch
from model import UNet
from diffusion import Diffusion
import matplotlib.pyplot as plt

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch_100.pt"))
    
    diffusion = Diffusion(device=device)
    samples = diffusion.sample(model, (16, 3, 32, 32))
    
    # Sauvegarder les images
    torch.save(samples, "generated/samples.pt")
    
    # Visualiser
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((samples[i].cpu().permute(1, 2, 0) + 1) / 2)
        plt.axis("off")
    plt.savefig("generated/samples.png")

if __name__ == "__main__":
    main()