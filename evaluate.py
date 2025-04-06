import torch
from diffusion import Diffusion
from model import UNet
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torch_fidelity import calculate_metrics
import torchvision.transforms as transforms

# Fonction de conversion en uint8, définie au niveau global pour être picklable
def to_uint8(x):
    return (x * 255).round().to(torch.uint8)

# Dataset personnalisé pour les images générées (format uint8 requis)
class GeneratedDataset(Dataset):
    def __init__(self, tensor_images):
        # Convertir les images générées [0,1] → [0,255] et en uint8
        self.images = (tensor_images * 255).clamp(0, 255).to(torch.uint8).cpu()

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx]  # Tenseur (3, 32, 32) en uint8

# Wrapper pour ne garder que les images du dataset CIFAR10
class ImageOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img  # On suppose que 'img' est déjà au format uint8

# Fonction pour générer des échantillons par batch
def batched_sample(diffusion, model, total_samples, batch_size, shape, device):
    samples_list = []
    for i in range(0, total_samples, batch_size):
        current_batch = min(batch_size, total_samples - i)
        batch_samples = diffusion.sample(model, (current_batch, *shape), device)
        samples_list.append(batch_samples)
        torch.cuda.empty_cache()
    return torch.cat(samples_list, dim=0)

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger le modèle entraîné
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch_20.pt", map_location=device))
    model.eval()

    # Initialiser l'objet diffusion
    diffusion = Diffusion(T=1000, beta_schedule='cosine', device=device)

    # Paramètres d'échantillonnage
    num_samples = 10   # Ajustez ce nombre selon vos besoins
    batch_size = 2
    shape = (3, 32, 32)

    # Génération des images par batch
    samples = batched_sample(diffusion, model, num_samples, batch_size, shape, device)
    samples = (samples + 1) / 2   # Passage dans [0,1]
    samples = samples.cpu()       # S'assurer que les images sont sur CPU

    # Conversion en Dataset pour torch-fidelity (format uint8 requis)
    generated_dataset = GeneratedDataset(samples)

    # Préparation du dataset de référence CIFAR10 (test)
    # On applique une transformation pour convertir les images en tenseurs uint8
    ref_transform = transforms.Compose([
        transforms.ToTensor(),   # Convertit en float [0,1]
        to_uint8                 # Convertit en uint8 [0,255]
    ])
    raw_cifar10 = CIFAR10(root='./data', train=False, download=True, transform=ref_transform)
    ref_dataset = ImageOnlyDataset(raw_cifar10)

    # Calcul des métriques via torch-fidelity
    metrics = calculate_metrics(
        input1=generated_dataset,
        input2=ref_dataset,
        fid=True,
        isc=True,
        cuda=torch.cuda.is_available()
    )

    print(f"\n✅ FID: {metrics['frechet_inception_distance']:.2f}")
    print(f"✅ Inception Score: {metrics['inception_score_mean']:.2f}")

if __name__ == "__main__":
    evaluate()
