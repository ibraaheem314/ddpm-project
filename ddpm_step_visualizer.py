import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model import UNet
from diffusion import Diffusion
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
MODEL_PATH = 'model_epoch_100.pth'
SAVE_DIR = 'step_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

# Correction de la normalisation pour 3 canaux (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation pour 3 canaux
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
x0, _ = dataset[10]  # Choix d'une image d'entrée
x0 = x0.unsqueeze(0).to(DEVICE)  # Ajout d'une dimension batch

# Initialisation du modèle avec les bonnes dimensions
model = UNet(in_channels=3, out_channels=3, image_size=32).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

diffusion = Diffusion(T=1000, beta_schedule="cosine", device=DEVICE)  # Définition de T et du schéma beta

# Choix d'un niveau de bruit (t entre 1 et T)
t = torch.tensor([100], device=DEVICE)

with torch.no_grad():
    # Étape 1 : Enregistrement de l'image originale
    x_orig = (x0 + 1) / 2  # Dés-normalisation
    save_image(x_orig, f"{SAVE_DIR}/1_original.png")

    # Étape 2 : Génération de l'image bruitée
    xt, noise = diffusion.forward_process(x0, t-1)  # forward_process génère xt et le bruit
    save_image((xt + 1) / 2, f"{SAVE_DIR}/2_noisy_t{t.item()}.png")

    # Étape 3 : Débruitage progressif
    x = xt.clone()  # Copie de l'image bruitée pour éviter les modifications en place
    for current_t in range(t.item(), 0, -1):  # Boucle depuis t jusqu'à 1
        t_tensor = torch.tensor([current_t], device=DEVICE)
        predicted_noise = model(x, t_tensor)
        x = diffusion._reverse_step(x, predicted_noise, current_t - 1)  # Utilisation directe de _reverse_step
    x_denoised = x

    save_image((x_denoised + 1) / 2, f"{SAVE_DIR}/3_denoised_t{t.item()}.png")

# Visualisation des images
image_paths = [
    f"{SAVE_DIR}/1_original.png",
    f"{SAVE_DIR}/2_noisy_t{t.item()}.png",
    f"{SAVE_DIR}/3_denoised_t{t.item()}.png"
]
titles = ["Original", f"Bruité (t={t.item()})", "Dénormé"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, (path, title) in enumerate(zip(image_paths, titles)):
    img = plt.imread(path)
    axs[i].imshow(img)
    axs[i].set_title(title)
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# Calcul des métriques
mse = F.mse_loss(x_denoised, x0).item()

x0_np = x0.squeeze().permute(1, 2, 0).cpu().numpy()
xd_np = x_denoised.squeeze().permute(1, 2, 0).cpu().numpy()

score = ssim(x0_np, xd_np, data_range=1.0, channel_axis=2)
print(f"MSE : {mse:.6f}")
print(f"SSIM : {score:.4f}")






# Ajoutez ces imports :
from torch_fidelity import calculate_metrics

# Après la génération de x_denoised :
# Convertissez les tenseurs en numpy arrays
x0_np = x0.squeeze().permute(1, 2, 0).cpu().numpy()
x_denoised_np = x_denoised.squeeze().permute(1, 2, 0).cpu().numpy()

# Calculez FID et IS (nécessite un dataset de référence)
metrics = calculate_metrics(
    input1=torch.stack([x_denoised.squeeze()]),
    input2="datasets/cifar10/val",
    cuda=True
)
print(f"FID: {metrics['frechet_inception_distance']:.2f}")
print(f"Inception Score: {metrics['inception_score_mean']:.2f}")