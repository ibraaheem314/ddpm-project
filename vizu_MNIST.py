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
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
import torch
import numpy as np
import random

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # pour une reproductibilité stricte
    torch.backends.cudnn.benchmark = False

seed_everything(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'checkpoints/mnist_epoch_180XXL.pt'
SAVE_DIR = 'step_outputs_mnist'
os.makedirs(SAVE_DIR, exist_ok=True)

# Chargement du dataset MNIST (test), normalisé
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x0 = torch.stack([dataset[i][0] for i in range(100)])  # 100 premières images
x0 = x0.to(DEVICE)

# Initialisation du modèle
model = UNet(
    in_channels=1,
    out_channels=1,
    model_channels=64,     
    channel_mult=(1, 2, 4),      
    num_res_blocks=1
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Initialisation de l'objet Diffusion
diffusion = Diffusion(T=1000, beta_schedule="cosine", device=DEVICE)

# Choix d'un niveau de bruit
t_level = 750
t = torch.tensor([t_level], device=DEVICE)

with torch.no_grad():
    # Image originale
    x_orig = (x0 + 1) / 2
    save_image(x_orig, f"{SAVE_DIR}/1_original.png")

    # Bruitage : forward_process
    xt, noise = diffusion.forward_process(x0, t - 1)
    save_image((xt + 1) / 2, f"{SAVE_DIR}/2_noisy_t{t_level}.png")

    # Débruitage progressif : on stocke la MSE et la SSIM à chaque étape
    x = xt.clone()
    mses = []
    ssims = []

    # On calcule la SSIM/MSE par rapport à x0 à chaque itération
    for current_t in range(t_level, 0, -1):
        mse_val = F.mse_loss(x, x0).item()

        x_np = x.squeeze().cpu().numpy()
        x0_np = x0.squeeze().cpu().numpy()
        ssim_val = ssim(x0_np[0], x_np[0], data_range=2.0)
        
        mses.append(mse_val)
        ssims.append(ssim_val)

        # Reverse step
        t_tensor = torch.tensor([current_t], device=DEVICE)
        predicted_noise = model(x, t_tensor)
        x = diffusion._reverse_step(x, predicted_noise, current_t)

    x_denoised = x
    save_image((x_denoised + 1) / 2, f"{SAVE_DIR}/3_denoised_t{t_level}.png")

# Calcul final de la MSE / SSIM
final_mse = F.mse_loss(x_denoised, x0).item()
x_denoised_np = x_denoised.squeeze().cpu().numpy()
final_ssim = ssim(x0_np[0], x_denoised_np[0], data_range=2.0)
print(f"MSE final: {final_mse:.6f}")
print(f"SSIM final: {final_ssim:.4f}")

# Visualisation + tracé des graphes
image_paths = [
    f"{SAVE_DIR}/1_original.png",
    f"{SAVE_DIR}/2_noisy_t{t_level}.png",
    f"{SAVE_DIR}/3_denoised_t{t_level}.png"
]
titles = ["Original", f"Bruité (t={t_level})", "Débruité"]

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
for i, (path, title) in enumerate(zip(image_paths, titles)):
    img = plt.imread(path)
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(title)
    axs[i].axis('off')
plt.tight_layout()
plt.show()


# Courbes de MSE et SSIM au fil du reverse process
steps = list(range(t_level, 0, -1))
data = pd.DataFrame({
    "Step": steps,
    "MSE": mses,
    "SSIM": ssims
})


plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
sns.lineplot(data=data, x="Step", y="MSE", linewidth=2)
plt.gca().invert_xaxis()
plt.title("MSE par rapport à x₀", fontsize=13)
plt.xlabel("Étape de reverse (t)")
plt.ylabel("MSE")

plt.subplot(1, 2, 2)
sns.lineplot(data=data, x="Step", y="SSIM", linewidth=2, color='orange')
plt.gca().invert_xaxis()
plt.title("SSIM par rapport à x₀", fontsize=13)
plt.xlabel("Étape de reverse (t)")
plt.ylabel("SSIM")

plt.tight_layout()
plt.savefig("step_outputs/SSIM_MSE180tri750.png", dpi=300)
plt.show()







import seaborn as sns
import matplotlib.pyplot as plt

# Données
epochs = [10, 20, 40, 50, 70, 90, 100, 120, 130, 150, 180]
mse = [0.0262, 0.0242, 0.0235, 0.0234, 0.0197, 0.0197, 0.0190, 0.0198, 0.0189, 0.0187, 0.0184]
ssim = [0.8791, 0.9595, 0.9806, 0.9848, 0.9861, 0.9879, 0.9880, 0.9883, 0.9879, 0.9883, 0.9884]

# Style Seaborn
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# Tracé MSE
plt.figure(figsize=(8, 4))
sns.lineplot(x=epochs, y=mse, marker='o', label='MSE')
plt.title("Évolution de la MSE selon le nombre d'époques")
plt.xlabel("Épochs")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("plots/mse_vs_epochs_seaborn.png", dpi=300)
plt.close()

# Tracé SSIM
plt.figure(figsize=(8, 4))
sns.lineplot(x=epochs, y=ssim, marker='o', color='orange', label='SSIM')
plt.title("Évolution du SSIM selon le nombre d'époques")
plt.xlabel("Épochs")
plt.ylabel("SSIM")
plt.tight_layout()
plt.savefig("plots/ssim_vs_epochs_seaborn.png", dpi=300)
plt.close()
