import torch
from diffusion import Diffusion
from model import UNet
from torch_fidelity import calculate_metrics

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load("checkpoints/epoch_1000.pt"))
    diffusion = Diffusion(T=1000)

    # Génération d'échantillons
    samples = diffusion.sample(model, (64, 3, 32, 32), device)
    samples = (samples + 1) / 2  # Dés-normalisation
    
    # Calcul des métriques
    metrics = calculate_metrics(
        input1=samples,
        input2="datasets/cifar10/val",
        cuda=True
    )
    
    print(f"FID: {metrics['frechet_inception_distance']:.2f}")
    print(f"Inception Score: {metrics['inception_score_mean']:.2f}")

if __name__ == "__main__":
    evaluate()