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
    sample_id = np.random.randint(100)
    lying_agent_label = 2
    judge_path = None
    dataset = "mnist"
    rollouts = 1000
    truth_agent = 0


@ex.automain
def run(N_to_mask, sample_id, lying_agent_label, judge_path, dataset, rollouts, index_of_truth_agent):
    if judge_path:
        path = judge_path
    elif dataset:
        path = "saved_models/" + dataset + str(N_to_mask)
    else:
        raise Exception("dataset must be specified")

    if dataset == "mnist":
        judge = MNISTJudge(N_to_mask=N_to_mask, model_dir=path)
    elif dataset == "fashion":
        judge = FashionJudge(N_to_mask=N_to_mask, model_dir=path)
    else:
        raise Exception("Unknown dataset in " + "dataset.txt: " + dataset)

    sample = judge.eval_data[sample_id].flatten()

    label = judge.eval_labels[sample_id]
    if lying_agent_label==label:
        lying_agent_label = (lying_agent_label + 1) % 10

    agent_lie = Agent(precommit_label=lying_agent_label, agentStrength=rollouts)
    agent_truth = Agent(precommit_label=label, agentStrength=rollouts)

    if index_of_truth_agent == 0:
        debate = Debate((agent_truth, agent_lie), judge, N_to_mask, sample, debug=False)
    else:
        debate = Debate((agent_lie, agent_truth), judge, N_to_mask, sample, debug=False)

    utility = debate.play()
    if utility==1:
        print("Winner: liar")
    elif utility==0:
        print("Draw")
    elif utility==-1:
        print("Winner: truth")
    else:
        print("Utility of the honest agent: ", utility*(-1)," (1=win, -1 = loss)")if truth_agent == winner:
