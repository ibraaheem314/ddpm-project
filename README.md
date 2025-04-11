# Projet Génératif avec Modèles de Diffusion (DDPM) sur MNIST

Ce dépôt illustre l’implémentation d’un modèle de diffusion probabiliste (DDPM) pour générer et reconstruire des images MNIST. Il s’inspire des travaux de **Ho et al. (2020)** sur les Denoising Diffusion Probabilistic Models.

## Structure du Projet

```bash
Projet_Generatif/
├── checkpoints/            # Sauvegarde des modèles (fichiers .pt)
│   ├── mnist_epoch_XX.pt
├── data/                   # Dossier où le dataset MNIST est téléchargé
│   └── MNIST/
├── diffusion.py            # Processus forward/backward (bruitage et débruitage)
├── model.py                # Définition du U-Net léger
├── train_MNIST.py          # Script pour entraîner le modèle
├── eval_MNIST.py           # Évaluation : FID, IS, MSE, SSIM
├── vizu_MNIST.py           # Visualisations détaillées du reverse process
└── step_outputs_mnist/     # Images résultantes (exemples bruités/débruités)
```

## Prérequis

1. **Environnement Python 3.8+** : un environnement virtuel est recommandé.
2. **GPU avec CUDA** : idéalement 8 Go de VRAM pour des batch size raisonnables.
3. **Bibliothèques** :
   - `torch`, `torchvision`, `tqdm`
   - `numpy`, `matplotlib`, `scikit-image` (pour le calcul SSIM)
   - `torch_fidelity` (pour FID/IS)
4. Installation rapide :
   ```bash
   pip install torch torchvision numpy matplotlib scikit-image torch_fidelity tqdm
   ```

## Scripts et Fonctionnalités

### 1. `model.py`
Définit l’architecture **U-Net** simplifiée :
- Entrées et sorties unicanales pour MNIST.
- Blocs résiduels (`ResidualBlock`) et éventuelle attention.
- Paramètres configurables (channel_mult, model_channels, etc.).

### 2. `diffusion.py`
- Implémente le **processus de diffusion** (ajout progressif de bruit) et le **reverse process** (débruitage) façon DDPM.
- Plusieurs paramètres : `T=1000`, `beta_schedule='cosine'`, etc.
- La fonction `_reverse_step` suit la postérieure gaussienne indiquée par Ho et al.

### 3. `train_MNIST.py`
- Charge MNIST (version train), normalise en `[ -1, 1 ]`.
- Définit un U-Net léger (64 ou 128 canaux) + l’objet Diffusion.
- Entraîne sur un certain nombre d’époques (ex. 100), calcule la loss MSE.
- Sauvegarde périodiquement le modèle dans `checkpoints/`.

**Lancement** :
```bash
python train_MNIST.py
```

### 4. `eval_MNIST.py`
- Évalue la performance du modèle sur MNIST (test) en générant un lot d’images.
- Calcule **FID** et **Inception Score** via `torch_fidelity`.
- Peut aussi mesurer MSE/SSIM en comparant des images débruitées à leur version originale.

**Lancement** :
```bash
python eval_MNIST.py
```

### 5. `vizu_MNIST.py`
- Permet d’examiner le reverse process étape par étape :
  1. Bruite une image MNIST en `$t`.
  2. Débruite itérativement de `$t` jusqu’à 0.
- Trace la **MSE** et la **SSIM** à chaque sous-étape.
- Sauvegarde les images (original, bruitée, débruitée) dans `step_outputs_mnist/`.

**Lancement** :
```bash
python vizu_MNIST.py
```

## Points Clés et Observations

- **Stabilité d’entraînement** : Rapidement (dès ~30–50 époques), la perte MSE descend sous 0.05.
- **Qualité de reconstruction** : SSIM > 0.98 après ~70–100 époques.
- **FID / IS** : Pertinents pour juger la distribution globale des échantillons. On obtient FID < 300 et IS ~ 2.5–3.0 avec un U-Net léger.
- **Limites** : Si le bruit initial est trop élevé (ex. t=999), la reconstruction échoue (MSE ~ 0.47, SSIM < 0.03).
- **Coût Computationnel** : T=1000 reste coûteux pour générer des images, mais faisable sur un GPU 8 Go.

## Pistes d'Amélioration
- **Dataset plus riche** : Tester sur CIFAR-10, CelebA.
- **Architecture plus profonde** : channel_mult=(1,2,2,2), ou plus de blocs résiduels.
- **Réduction du nombre d’étapes** : Approches type DDIM pour accélérer la génération.
- **Optimisation** : Mixed Precision (AMP), meilleur scheduler LR.

## Références
- \textbf{Jonathan Ho}, Ajay Jain, Pieter Abbeel. \emph{Denoising Diffusion Probabilistic Models}. NeurIPS, 2020.

