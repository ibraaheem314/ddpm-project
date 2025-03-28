import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model import UNet
from diffusion import Diffusion
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64  # Plus c'est bas plus y a erreur de reconstruire l'image
MODEL_PATH = 'model_epoch_100.pth'
SAVE_DIR = 'step_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
x0, _ = dataset[10]  # une seule image
x0 = x0.unsqueeze(0).to(DEVICE)


model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

diffusion = Diffusion(device=DEVICE)

# Choisir un t (niveau de bruit)
t = torch.tensor([100], device=DEVICE)

# Étapes :
with torch.no_grad():
    x_orig = (x0 + 1) / 2
    save_image(x_orig, f"{SAVE_DIR}/1_original.png")

    noise = torch.randn_like(x0)
    x_noisy = diffusion.q_sample(x0, t, noise)
    save_image((x_noisy + 1) / 2, f"{SAVE_DIR}/2_noisy_t{t.item()}.png")

    x = x_noisy
    for current_t in reversed(range(t.item())):
        x = diffusion.p_sample(model, x, torch.tensor([current_t], device=DEVICE))
        x_denoised = x
    save_image((x_denoised + 1) / 2, f"{SAVE_DIR}/3_denoised_t{t.item()}.png")


image_paths = [
    f"{SAVE_DIR}/1_original.png",
    f"{SAVE_DIR}/2_noisy_t{t.item()}.png",
    f"{SAVE_DIR}/3_denoised_t{t.item()}.png"
]

titles = ["Original", "Noisy", "Denoised"]

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    img = plt.imread(image_paths[i])
    axs[i].imshow(img)
    axs[i].set_title(titles[i])
    axs[i].axis('off')
plt.tight_layout()
plt.show()


# Métriques
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

mse = F.mse_loss(x_denoised, x0).item()

x0_np = x0.squeeze().permute(1, 2, 0).cpu().numpy()
xd_np = x_denoised.squeeze().permute(1, 2, 0).cpu().numpy()

score = ssim(x0_np, xd_np, data_range=1.0, channel_axis=2)

print(f"MSE between denoised and original: {mse:.6f}")
print(f"SSIM between denoised and original: {score:.4f}")
