import numpy as np
from judge import MNISTJudge

if __name__ == "__main__":
    judge = MNISTJudge(N_pixels=4)
    img = judge.eval_data[0]
    img_flat = np.reshape(img, img.shape[0] * img.shape[1])
    nonzero = np.where(img_flat > 0)[0]
    idx = np.random.choice(np.array(nonzero), 4)
    mask_flat = np.zeros_like(img_flat)
    mask_flat[idx] = 1
    mask = np.reshape(mask_flat, img.shape)
    judge.train(10)
    judge.evaluate_debate(img, mask, 1, 3)
