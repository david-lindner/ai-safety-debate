import matplotlib.pyplot as plt
import numpy as np

def plot_image_mask(state, img_size, filename=None):
    """
    Visualizes a debate state, in case the sample is an image.

    Args:
    state: DebateState
    img_size: width / height or the (square) image
    """
    plt.clf()
    image = np.reshape(state.sample, (img_size, img_size))
    mask = np.reshape(state.mask, (img_size, img_size))
    zeros = np.zeros_like(mask)
    alpha = 0.7
    mask_rgba = np.stack((mask, zeros, zeros, mask*alpha), axis=2)
    plt.imshow(image, cmap="gist_gray")
    plt.imshow(mask_rgba)
    plt.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
