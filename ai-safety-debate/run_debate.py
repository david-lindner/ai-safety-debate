import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import Agent

ex = Experiment("mnist_debate")
ex.observers.append(FileStorageObserver.create("experiments"))


@ex.config
def cfg():
    N_to_mask = 4
    sample_id = 0
    lying_agent_label = 2
    judge_path = None
    dataset = None
    rollouts = 1000


@ex.automain
def run(N_to_mask, sample_id, lying_agent_label, judge_path, dataset, rollouts):
    if judge_path:
        path = judge_path
    elif dataset:
        path = "saved_models/" + dataset + str(N_to_mask)
    else:
        raise Exception("Either judge_path or dataset needs to be specified")

    if dataset == "mnist":
        judge = MNISTJudge(N_to_mask=N_to_mask, model_dir=path)
    elif dataset == "fashion":
        judge = FashionJudge(N_to_mask=N_to_mask, model_dir=path)
    else:
        raise Exception("Unknown dataset in " + "dataset.txt: " + dataset)

    sample = judge.eval_data[sample_id].flatten()

    label = judge.eval_labels[sample_id]
    assert label != lying_agent_label

    agent1 = Agent(precommit_label=lying_agent_label, agentStrength=rollouts)
    agent2 = Agent(precommit_label=label, agentStrength=rollouts)

    debate = Debate((agent1, agent2), judge, N_to_mask, sample)

    winner = debate.play()
    print("Winner", winner)
