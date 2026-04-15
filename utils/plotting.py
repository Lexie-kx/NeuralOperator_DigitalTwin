import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(gt_field, pred_field, save_path="pod_eval_result.png"):
    plt.figure(figsize=(18, 5))
    gt_u, pred_u = gt_field[:, 0].reshape(64, 64), pred_field[:, 0].reshape(64, 64)
    error_u = np.abs(gt_u - pred_u)

    for i, (data, title, cmap) in enumerate(
            [(gt_u, "Ground Truth", "jet"), (pred_u, "Prediction", "jet"), (error_u, "Abs Error", "coolwarm")]):
        plt.subplot(1, 3, i + 1)
        plt.title(title)
        im = plt.imshow(data, cmap=cmap, origin='lower')
        plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()