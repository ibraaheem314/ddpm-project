import torch
from diffusion import Diffusion
from model import UNet
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Charger le modèle avec la configuration attendue (modifie les paramètres si besoin)
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch_100.pt", map_location=device))
    model.eval()
    
    # Initialiser l'objet diffusion
    diffusion = Diffusion(T=1000, beta_schedule='cosine', device=device)
    
    # Génération d'échantillons (ici, 10 000 images)
    num_samples = 10000
    samples = diffusion.sample(model, (num_samples, 3, 32, 32), device)
    samples = (samples + 1) / 2  # Dés-normalisation vers [0, 1]
    
    # Charger le jeu de test CIFAR-10 (images jamais vues lors de l'entraînement)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    metrics = calculate_metrics(
        input1=samples,
        input2=test_loader,
        cuda=True
    )
    
    print(f"FID: {metrics['frechet_inception_distance']:.2f}")
    print(f"Inception Score: {metrics['inception_score_mean']:.2f}")

if __name__ == "__main__":
    evaluate()
