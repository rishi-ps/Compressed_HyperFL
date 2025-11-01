import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11
})

def load_if_exists(filename):
    return np.load(filename) if os.path.exists(filename) else None


def plot_training_loss():
    """Plot training loss vs global rounds for all methods."""
    methods = ["FedAvg", "HyperFL", "HyperFL_Sparse"]
    styles = {
        "FedAvg": ("-", "tab:blue"),
        "HyperFL": ("--", "tab:orange"),
        "HyperFL_Sparse": (":", "tab:red"),
    }

    plt.figure(figsize=(7, 5))
    for method in methods:
        loss_file = f"{method}_train_loss.npy"
        if os.path.exists(loss_file):
            loss = np.load(loss_file)
            if len(loss) > 3:  # smooth only for longer runs
                loss = gaussian_filter1d(loss, sigma=1)
            plt.plot(range(1, len(loss) + 1), loss,
                     linestyle=styles[method][0],
                     color=styles[method][1],
                     linewidth=2,
                     label=method)

    plt.xlabel("Training Round")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Global Rounds")
    plt.ylim(0, 5)  # scale like the paper
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=300)
    plt.close()
    print(" Saved: loss_comparison.png")


def plot_parameter_difference():
    """
    Optional: visualize round-to-round parameter changes (like paper Fig. b).
    Requires saved model checkpoints as .pt files named:
    model_round_X.pt for each round.
    """
    ckpts = sorted([f for f in os.listdir('.') if f.startswith("model_round_") and f.endswith(".pt")])
    if len(ckpts) < 2:
        print("  No model_round_X.pt files found — skipping parameter-difference plot.")
        return

    diffs = []
    prev_params = None

    for ckpt in ckpts:
        model = torch.load(ckpt, map_location='cpu')
        params = torch.cat([v.flatten() for v in model.values()])
        if prev_params is not None:
            diffs.append(torch.norm(params - prev_params).item())
        prev_params = params

    diffs = np.array(diffs)
    rounds = np.arange(1, len(diffs) + 1)
    diffs = gaussian_filter1d(diffs, sigma=1)

    plt.figure(figsize=(7, 5))
    plt.plot(rounds, diffs, color='green', linewidth=2)
    plt.xlabel("Training Round")
    plt.ylabel("Parameter Difference")
    plt.title("Round-to-Round Parameter Difference")

    # inset zoom (paper-like)
    ax = plt.gca()
    inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
    inset.plot(rounds, diffs, color='red', linewidth=1.5)
    inset.set_xlim(0, min(200, len(rounds)))
    inset.set_ylim(0, diffs.max() * 1.05)
    inset.set_title("Zoomed In", fontsize=9)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("parameter_difference.png", dpi=300)
    plt.close()
    print(" Saved: parameter_difference.png")


def plot_sparsity():
    """Plot sparsity curve for HyperFL_Sparse (optional)."""
    filename = "HyperFL_Sparse_sparsity.npy"
    if not os.path.exists(filename):
        print("  No sparsity log found — skipping sparsity plot.")
        return

    sparsity = np.load(filename)
    rounds = np.arange(1, len(sparsity) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(rounds, sparsity, color='purple', linewidth=2)
    plt.xlabel("Global Rounds")
    plt.ylabel("Sparsity (%)")
    plt.title("Gradient Sparsity Across Rounds")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("sparsity_curve.png", dpi=300)
    plt.close()
    print(" Saved: sparsity_curve.png")


if __name__ == "__main__":
    plot_training_loss()
    plot_parameter_difference()
    plot_sparsity()
    print("\nAll plots saved successfully in current directory.")
