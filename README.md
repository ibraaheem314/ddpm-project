# Projet Génératif avec Diffusion Probabiliste (DDPM) – MNIST

Ce dépôt propose une implémentation d’un modèle de diffusion (Denoising Diffusion Probabilistic Model) pour générer et reconstruire des images MNIST. L’implémentation s’appuie sur l’architecture décrite par Ho et al. (NeurIPS 2020), adaptée aux contraintes matérielles modérées.

---

## Arborescence du Projet

```bash
Projet_Generatif/
├── checkpoints/            # Sauvegarde des checkpoints (fichiers .pt)
│   ├── mnist_epoch_XX.pt
├── data/                   # Dossier où MNIST est téléchargé
│   └── MNIST/
├── diffusion.py            # Processus forward (bruit) et reverse (débruitage)
├── model.py                # Définition d’un U-Net léger (1 canal)
├── train_MNIST.py          # Script d’entraînement principal
├── eval_MNIST.py           # Scripts d’évaluation (FID, IS, MSE, SSIM)
├── vizu_MNIST.py           # Visualisation approfondie du reverse step
└── step_outputs_mnist/     # Résultats intermédiaires (images, courbes)
```

## Prérequis

1. **Environnement Python 3.8+** (idéalement).
2. **GPU CUDA** (8 Go de VRAM recommandé).
3. **Bibliothèques** : `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-image`, `torch_fidelity`, `tqdm`.

Installation express :
```bash
pip install torch torchvision numpy matplotlib scikit-image torch_fidelity tqdm
```

---

## Scripts Principaux

### `model.py`
- Définit un **U-Net** spécialement conçu pour MNIST (monocanal).
- Inclus des blocs résiduels (`ResidualBlock`) et éventuellement un bloc d’attention.
- Paramétrable (nombre de canaux, `channel_mult`, etc.).

### `diffusion.py`
- Gère la **logique de diffusion** (forward : ajout progressif de bruit) et le **reverse** (débruitage).
- Utilise la planification de bruitage cosinus (`beta_schedule='cosine'`) et un `_reverse_step` fidèle à la formulation Ho et al.

### `train_MNIST.py`
- Charge MNIST (train), normalise en `[ -1, 1 ]`.
- Met en place l’entraînement :
  - Création d’un U-Net léger.
  - Boucle sur `T=1000` étapes de diffusion.
  - Optimiseur AdamW.
- Sauvegarde des checkpoints dans `checkpoints/`.

**Exemple d’utilisation** :
```bash
python train_MNIST.py
```

### `eval_MNIST.py`
- Évalue les checkpoints sauvegardés :
  - Génère un batch d’images via `diffusion.sample(...)`.
  - Calcule la **FID** et l’**Inception Score** via `torch_fidelity`.
  - Possibilité de mesurer la **MSE** et la **SSIM** directement.

**Exemple d’utilisation** :
```bash
python eval_MNIST.py
```

### `vizu_MNIST.py`
- Propose une **visualisation pas à pas** du reverse process :
  1. Bruitage d’une (ou plusieurs) image(s) MNIST à l’instant `t`.
  2. Débruitage itératif en remontant de `t` jusqu’à `0`.
  3. Calcul/traçage de la **MSE** et de la **SSIM** à chaque étape.
- Enregistre tous les résultats (images, courbes) dans `step_outputs_mnist/`.

**Exemple d’utilisation** :
```bash
python vizu_MNIST.py
```

---

## Points Forts et Limites

- **Simplicité et Rapidité** : Sur MNIST, le U-Net léger converge vite (~30–50 époques) pour des reconstructions de qualité (SSIM>0.98).
- **Scores Globaux** : FID autour de 220–300 et IS ~2.5–3.0 après ~40–70 époques, selon la taille du batch généré.
- **Limites** : si le bruit initial est trop élevé (ex. `t=999`), la reconstruction s’effondre (MSE~0.47, SSIM<0.03). Un T=1000 complet est aussi plus coûteux en temps.
- **Corriger `_reverse_step()`** : Étape cruciale pour éviter les artefacts de type « code QR ».

---

## Pistes d’Évolution

1. **Datasets plus complexes** : CIFAR-10, CelebA, LSUN pour confirmer la robustesse du DDPM.
2. **Architecture plus large** : augmenter `model_channels`, multiplier les blocs résiduels.
3. **Réduction de T** : Approches DDIM pour accélérer l’échantillonnage.
4. **Recherche d’hyperparamètres** : scheduling du LR, Mixed Precision plus poussé, etc.

---

## Références
- Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- [Diffusion Models GitHub (Référence Ho)](https://github.com/hojonathanho/diffusion)

---

_Projet réalisé dans un contexte d’apprentissage génératif : MSE / SSIM / FID / IS illustrent la qualité des reconstructions et la vraisemblance statistique. N’hésitez pas à ouvrir une _issue_ ou proposer un _pull request_ si vous souhaitez contribuer !_