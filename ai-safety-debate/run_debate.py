import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge
from debate import Debate
from agent import Agent

ex = Experiment("mnist_debate")
ex.observers.append(FileStorageObserver.create("experiments"))


@ex.config
def cfg():
    N_to_mask = 4
    img_id = 0
    lying_agent_label = 2
    judge_path = None


@ex.automain
def run(N_to_mask, img_id, lying_agent_label, judge_path):
    N_to_mask = 4

    if not judge_path:
        raise Exception("No judge given. Use train_judge.py first.")

    judge = MNISTJudge(N_to_mask=N_to_mask, restore_model_from=judge_path)

    img = judge.eval_data[img_id]
    img_flat = np.reshape(img, img.shape[0] * img.shape[1])
    label = judge.eval_labels[img_id]
    assert label != lying_agent_label

    agent1 = Agent(precommit_label=lying_agent_label)
    agent2 = Agent(precommit_label=label)

    debate = Debate((agent1, agent2), judge, N_to_mask, img_flat)
    winner = debate.play()
    print("Winner", winner)
