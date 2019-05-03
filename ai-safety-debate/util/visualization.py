import matplotlib.pyplot as plt
import numpy as np


def plot_image_mask(state, filename=None):
    """
    Visualizes a debate state, in case the sample is an image.

    Args:
    state: DebateState
    filename: if given, plot will be written to a file instead of shown
    """
    plt.clf()
    img_size = np.sqrt(state.mask[0].size)
    assert img_size % 1 == 0
    img_size = int(img_size)
    image = np.reshape(state.sample, (img_size, img_size))
    # first player mask
    mask_1 = np.reshape(state.mask[0], (img_size, img_size))
    # second player mask
    mask_2 = np.reshape(state.mask[1], (img_size, img_size))
    zeros = np.zeros_like(mask_1)
    alpha = 0.7
    # red overlay
    mask_1_rgba = np.stack((mask_1, zeros, zeros, mask_1 * alpha), axis=2)
    # blue overlay
    mask_2_rgba = np.stack((zeros, zeros, mask_2, mask_2 * alpha), axis=2)
    plt.imshow(image, cmap="gist_gray")
    plt.imshow(mask_1_rgba)
    plt.imshow(mask_2_rgba)
    plt.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
