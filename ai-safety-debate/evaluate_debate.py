import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver
from random import randint

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import Agent

ex = Experiment("mnist_debate_eval")
ex.observers.append(FileStorageObserver.create("experiments"))


@ex.config
def cfg():
    N_to_mask = 4
    judge_path = './mnist_judge'
    dataset = 'mnist'
    dataset_size = 10


@ex.automain
def run(N_to_mask, judge_path, dataset, dataset_size):
    if judge_path:
        path = judge_path
    elif dataset:
        path = "saved_models/" + dataset + str(N_to_mask)
    else:
        raise Exception("Either judge_path or dataset needs to be specified")

    if dataset == 'mnist':
        judge = MNISTJudge(N_to_mask=N_to_mask, restore_model_from=path)
    elif dataset == 'fashion':
        judge = FashionJudge(N_to_mask=N_to_mask, restore_model_from=path)
    else:
        raise Exception("Unknown dataset in " + "dataset.txt: " + dataset)

    if not dataset_size:
        dataset_size = len(judge.eval_data)

    lying_agent_wins = 0
    truth_agent_wins = 0

    for sample_id in range(dataset_size):

        sample = judge.eval_data[sample_id].flatten()

        label = judge.eval_labels[sample_id]
        lying_agent_label = randint(0, 9)
        while (label == lying_agent_label):
        	lying_agent_label = randint(0, 9)

        agent1 = Agent(precommit_label=lying_agent_label)
        agent2 = Agent(precommit_label=label)

        debate = Debate((agent1, agent2), judge, N_to_mask, sample)
        winner = debate.play()

        if winner == 0:
        	lying_agent_wins = lying_agent_wins + 1
        	print(":(")
        else:
        	truth_agent_wins = truth_agent_wins + 1
        	print(":)")

    print("Truth wins", truth_agent_wins/dataset_size)
    print("Truth wins", truth_agent_wins, "out of", dataset_size)
