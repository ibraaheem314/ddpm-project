# Projet Génératif avec Diffusion Probabiliste (DDPM) – MNIST

Ce dépôt propose une implémentation d'un modèle de diffusion (Denoising Diffusion Probabilistic Model) pour générer et reconstruire des images MNIST. L'implémentation s'appuie sur l'architecture de Ho et al. (NeurIPS 2020), adaptée à des contraintes matérielles modérées.

---

## Arborescence du Projet

```bash
Projet_Generatif/
├── checkpoints/            # Sauvegarde des checkpoints (fichiers .pt)
│   ├── mnist_epoch_XX.pt
├── data/                   # Dossier où MNIST est téléchargé
│   └── MNIST/
├── diffusion.py            # Processus forward (bruit) et reverse (débruitage)
├── model.py                # Définition d’un U-Net léger (un canal)
├── train_MNIST.py          # Script d’entraînement principal
├── eval_MNIST.py           # Évaluation (FID, IS, MSE, SSIM)
├── vizu_MNIST.py           # Visualisation détaillée du reverse step
└── step_outputs_mnist/     # Résultats (images, courbes)
```

## Prérequis

1. Environnement Python 3.8 ou plus.
2. GPU avec CUDA (8 Go de VRAM recommandé).
3. Bibliothèques : torch, torchvision, numpy, matplotlib, scikit-image, torch_fidelity, tqdm.

Installation rapide :
```bash
pip install torch torchvision numpy matplotlib scikit-image torch_fidelity tqdm
```

---

## Scripts Principaux

### model.py
Définit un U-Net adapté à MNIST (monocanal). Inclut des blocs résiduels (ResidualBlock) et éventuellement un bloc d’attention. La taille du modèle peut être ajustée via model_channels ou channel_mult.

### diffusion.py
Implémente la logique de diffusion (ajout progressif de bruit) et de reverse (débruitage). Utilise un planning de bruitage cosinus (beta_schedule='cosine'), et une fonction _reverse_step conforme à Ho et al.

### train_MNIST.py
Charge MNIST (train), normalise les images en [-1, 1], instancie le modèle U-Net et l’objet Diffusion, puis lance l’entraînement. Les checkpoints sont sauvegardés dans checkpoints/.

Exemple :
```bash
python train_MNIST.py
```

### eval_MNIST.py
Évalue les checkpoints entraînés en générant un lot d’images via diffusion.sample(...), puis calcule les métriques FID et Inception Score avec torch_fidelity. Permet également de mesurer la MSE et la SSIM.

Exemple :
```bash
python eval_MNIST.py
```

### vizu_MNIST.py
Fournit une visualisation étape par étape du reverse process :
1. Bruitage d’une image MNIST à un instant t.
2. Débruitage progressif jusqu’à t=0.
3. Calcul et traçage de la MSE et de la SSIM à chaque étape.

Enregistre les images et graphes dans step_outputs_mnist/.

Exemple :
```bash
python vizu_MNIST.py
```

---

## Points Forts et Limites

- Simplicité et rapidité : le U-Net converge assez vite (~30–50 époques) sur MNIST.
- Scores globaux : FID autour de 220–300 et IS vers 2.5–3.0 après ~40–70 époques.
- Limites : si le bruit initial est trop élevé (t=999), la reconstruction échoue (MSE>0.4, SSIM<0.03). T=1000 augmente aussi le coût du sampling.
- Correction de _reverse_step : indispensable pour éviter des artefacts.

---

## Pistes d’Évolution

1. Datasets plus complexes (CIFAR-10, CelebA) pour valider la robustesse du DDPM.
2. Architecture plus large : augmenter model_channels, multiplier les blocs résiduels.
3. Réduction de T : approches DDIM pour accélérer l’échantillonnage.
4. Optimisation : scheduling du LR, Mixed Precision (AMP), etc.

---

## Références
- Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- [Diffusion Models GitHub (Référence Ho)](https://github.com/hojonathanho/diffusion)

---

Projet réalisé dans un contexte d’apprentissage génératif. MSE, SSIM, FID et IS permettent de juger la qualité des reconstructions et de la distribution générée. N’hésitez pas à ouvrir une issue ou proposer une pull request si vous souhaitez contribuer.
