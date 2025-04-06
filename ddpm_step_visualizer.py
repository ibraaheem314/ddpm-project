import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from model import UNet
from diffusion import Diffusion
from torch_fidelity import calculate_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'checkpoints/model_epoch_10.pt'
SAVE_DIR = 'step_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

# Préparation du dataset CIFAR-10 (test) avec normalisation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
x0, _ = dataset[10]  # Choix d'une image d'entrée de test
x0 = x0.unsqueeze(0).to(DEVICE)  # Ajout de la dimension batch

# Initialisation du modèle avec les paramètres corrects
model = UNet(
    in_channels=3,
    out_channels=3,
    model_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Initialisation de l'objet Diffusion
diffusion = Diffusion(T=1000, beta_schedule="cosine", device=DEVICE)

# Choix d'un niveau de bruit (par exemple t=100)
t_level = 100
t = torch.tensor([t_level], device=DEVICE)

with torch.no_grad():
    # Étape 1 : Sauvegarde de l'image originale (dés-normalisée)
    x_orig = (x0 + 1) / 2
    save_image(x_orig, f"{SAVE_DIR}/1_original.png")
    
    # Étape 2 : Génération de l'image bruitée via le forward process
    xt, noise = diffusion.forward_process(x0, t - 1)
    save_image((xt + 1) / 2, f"{SAVE_DIR}/2_noisy_t{t_level}.png")
    
    # Étape 3 : Débruitage progressif (reverse process)
    x = xt.clone()
    for current_t in range(t_level, 0, -1):
        t_tensor = torch.tensor([current_t], device=DEVICE)
        predicted_noise = model(x, t_tensor)
        x = diffusion._reverse_step(x, predicted_noise, current_t)
    x_denoised = x
    save_image((x_denoised + 1) / 2, f"{SAVE_DIR}/3_denoised_t{t_level}.png")

# Calcul des métriques d'erreur
mse_val = F.mse_loss(x_denoised, x0).item()
x0_np = x0.squeeze().permute(1, 2, 0).cpu().numpy()
x_denoised_np = x_denoised.squeeze().permute(1, 2, 0).cpu().numpy()
ssim_val = ssim(x0_np, x_denoised_np, data_range=1.0, channel_axis=2)
print(f"MSE: {mse_val:.6f}")
print(f"SSIM: {ssim_val:.4f}")

# Calcul des métriques FID et Inception Score
# Remarque : Pour un calcul robuste, il est préférable d'utiliser plusieurs images générées.
generated_sample = x_denoised.squeeze().unsqueeze(0)  # Ici, on utilise l'image débruitée comme exemple

metrics = calculate_metrics(
    input1=generated_sample,
    input2="datasets/cifar10/val",  # Assure-toi que ce dossier contient le jeu de test CIFAR-10
    cuda=torch.cuda.is_available(),
    isc = True,
    fid = True# Spécifie les métriques sous forme de liste
)
print(f"FID: {metrics.get('frechet_inception_distance', float('nan')):.2f}")
print(f"Inception Score: {metrics.get('inception_score_mean', float('nan')):.2f}")

# Visualisation des images sauvegardées
image_paths = [
    f"{SAVE_DIR}/1_original.png",
    f"{SAVE_DIR}/2_noisy_t{t_level}.png",
    f"{SAVE_DIR}/3_denoised_t{t_level}.png"
]
titles = ["Original", f"Bruité (t={t_level})", "Débruité"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, (path, title) in enumerate(zip(image_paths, titles)):
    img = plt.imread(path)
    axs[i].imshow(img)
    axs[i].set_title(title)
    axs[i].axis('off')
plt.tight_layout()
plt.show()
