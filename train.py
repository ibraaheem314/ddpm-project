import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from model import UNet
from diffusion import Diffusion
import torch.nn.functional as F

DEVICE = "cuda"
BATCH_SIZE = 128
LR = 2e-4
NUM_EPOCHS = 100
T = 1000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation pour CIFAR-10 (3 canaux)
])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(in_channels=3, out_channels=3, image_size=32).to(DEVICE)
diffusion = Diffusion(T=T, beta_schedule="cosine")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

class EMA:
    def __init__(self, beta=0.9999):
        self.beta = beta
        self.shadow = {}
    
    def register(self, model):
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name] = self.beta * self.shadow[name] + (1 - self.beta) * param.data

ema = EMA()
ema.register(model)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for x0, _ in tqdm(train_loader):
        x0 = x0.to(DEVICE)
        t = torch.randint(1, T+1, (x0.size(0,),), device=DEVICE)
        
        xt, noise = diffusion.forward_process(x0, t-1)
        
        with torch.autocast(device_type=DEVICE, enabled=True):
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        ema.update(model)
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")