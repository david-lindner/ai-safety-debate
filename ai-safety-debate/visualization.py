import matplotlib.pyplot as plt
import numpy as np

def plot_image_mask(state, img_size, filename=None):
    plt.clf()
    mask = state[0]
    image = state[1]
    image = np.reshape(image, (img_size, img_size))
    mask = np.reshape(mask, (img_size, img_size))
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
