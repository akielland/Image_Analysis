import matplotlib.pyplot as plt
import numpy as np

def plot_covariance_matrix(cov, classes):
    fig, ax = plt.subplots()
    im = ax.imshow(cov, cmap='coolwarm')

    # Set ticks for each class
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

    # Loop over data dimensions and create text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, f"{cov[i, j]:.2f}", ha="center", va="center", color="w")

    # Save figure as an image file
    fig.savefig('covariance_matrix.png', dpi=300, bbox_inches='tight')
