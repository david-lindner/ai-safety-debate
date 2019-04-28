import numpy as np

from judge import MNISTJudge
from debate import Debate
from agent import Agent

if __name__ == "__main__":
    N_pixels = 4

    judge = MNISTJudge(N_pixels=N_pixels)
    judge.train(100)

    img = judge.eval_data[0]
    img_flat = np.reshape(img, img.shape[0] * img.shape[1])
    label = judge.eval_labels[0]

    if label > 0:
        alt_label = label - 1
    else:
        alt_label = label + 1

    agent1 = Agent(precommit_label=label)
    agent2 = Agent(precommit_label=label + 1)

    debate = Debate((agent1, agent2), judge, N_pixels, img_flat)
    winner = debate.play()
    print("Winner", winner)
