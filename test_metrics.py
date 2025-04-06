from torch_fidelity import calculate_metrics
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# Une fausse image générée (batch 1, 3x32x32)
generated = torch.randn(1, 3, 32, 32)

# Dataset CIFAR-10 pour référence
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
ref_dataset = CIFAR10('./data', train=False, download=True, transform=transform)
ref_loader = DataLoader(ref_dataset, batch_size=32)

metrics = calculate_metrics(
    input1=generated,
    input2=ref_loader,
    fid=True,
    isc=True,
    cuda=torch.cuda.is_available()
)

print(metrics)
