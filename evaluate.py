# evaluate.py
from pytorch_fid import fid_score
from torch_fidelity import calculate_metrics

# Calcul de la FID
fid = fid_score.calculate_fid_given_paths(
    ["generated", "data/cifar10/test"],
    batch_size=50,
    device="cuda"
)
print(f"FID: {fid}")

# Calcul de l'Inception Score
metrics = calculate_metrics(
    input1="generated",
    isc=True,
    fid=False
)
print(f"Inception Score: {metrics['inception_score_mean']}")