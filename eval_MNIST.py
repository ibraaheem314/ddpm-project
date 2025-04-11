import torch
from diffusion import Diffusion
from model import UNet
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics
import torchvision.transforms as transforms

# Dataset pour les images générées
class GeneratedDataset(Dataset):
    def __init__(self, tensor_images):
        # Convertir [-1,1] -> [0,255] puis uint8
        self.images = ((tensor_images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu()

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        img = self.images[idx]
        return img.repeat(3, 1, 1)

# Dataset pour les vraies images de référence
class ImageOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = (img * 255).round().to(torch.uint8)
        return img.repeat(3, 1, 1)

# Génération batchée d'images
def batched_sample(diffusion, model, total_samples, batch_size, shape, device):
    samples_list = []
    for i in range(0, total_samples, batch_size):
        current_batch = min(batch_size, total_samples - i)
        batch_samples = diffusion.sample(model, (current_batch, *shape), device)
        samples_list.append(batch_samples)
        torch.cuda.empty_cache()
    return torch.cat(samples_list, dim=0)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=1,
        out_channels=1,
        model_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=1
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/mnist_epoch_100XXL.pt", map_location=device))
    model.eval()

    diffusion = Diffusion(T=1000, beta_schedule="cosine", device=device)

    # Génération d'images
    num_samples = 1000
    batch_size = 512
    shape = (1, 28, 28)

    with torch.no_grad():
        samples = batched_sample(diffusion, model, num_samples, batch_size, shape, device)

    # Dataset généré
    generated_dataset = GeneratedDataset(samples)

    # Dataset de référence
    ref_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    raw_mnist = FashionMNIST("./data", train=False, download=True, transform=ref_transform)
    ref_dataset = ImageOnlyDataset(raw_mnist)

    # Calcul des métriques
    metrics = calculate_metrics(
    input1=generated_dataset,
    input2='data/mnist_test',
    cuda=torch.cuda.is_available(),
    fid=True,
    isc=True)


    print(f"FID: {metrics['frechet_inception_distance']:.2f}")
    print(f"Inception Score: {metrics['inception_score_mean']:.2f}")

if __name__ == "__main__":
    evaluate()