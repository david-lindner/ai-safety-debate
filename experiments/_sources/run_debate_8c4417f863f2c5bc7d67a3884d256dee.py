import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge
from debate import Debate
from agent import Agent

ex = Experiment('mnist_debate')
ex.observers.append(FileStorageObserver.create('experiments'))

@ex.config
def cfg():
    N_pixels = 4
    img_id = 0
    lying_agent_label = 2

@ex.automain
def run(N_pixels, img_id, lying_agent_label):
    N_pixels = 4

    judge = MNISTJudge(N_pixels=N_pixels)
    # judge.train(100)  # TODO reactivate after debate judging is fixed

    img = judge.eval_data[img_id]
    img_flat = np.reshape(img, img.shape[0] * img.shape[1])
    label = judge.eval_labels[img_id]
    assert label != lying_agent_label

    agent1 = Agent(precommit_label=lying_agent_label)
    agent2 = Agent(precommit_label=label)

    debate = Debate((agent1, agent2), judge, N_pixels, img_flat)
    winner = debate.play()
    print("Winner", winner)
