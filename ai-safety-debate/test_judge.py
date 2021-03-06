"""
Used for testing the MNIST judge. Trains a judge for a (hardcoded) number of steps
and prints out its accuracy.
"""

import numpy as np
from judge import MNISTJudge

if __name__ == "__main__":
    judge = MNISTJudge(4)
    img = judge.eval_data[0]
    img_flat = np.reshape(img, img.shape[0] * img.shape[1])
    nonzero = np.where(img_flat > 0)[0]
    idx = np.random.choice(np.array(nonzero), 4)
    mask_flat = np.zeros_like(img_flat)
    mask_flat[idx] = 1
    N_steps = 100
    judge.train(N_steps)
    print(judge.evaluate_accuracy())
    print(judge.evaluate_debate(np.stack((mask_flat, img_flat * mask_flat)), [0, 2]))
