"""
Run an individual debate and plots an image and returns the results.
"""


import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import DebateAgent, DebatePlayers

ex = Experiment("run_debate")
ex.observers.append(FileStorageObserver.create("experiments"))


@ex.config
def cfg():
    N_to_mask = 4
    sample_id = np.random.randint(100)
    lying_agent_label = None
    judge_path = None
    dataset = "mnist"
    rollouts = 1000
    index_of_our_agent = 1
    binary_rewards = False
    changing_sides = False


@ex.automain
def run(
    N_to_mask,
    sample_id,
    lying_agent_label,
    judge_path,
    dataset,
    rollouts,
    index_of_our_agent,
    binary_rewards,
    changing_sides,
):
    assert index_of_our_agent in [0, 1]

    if judge_path:
        path = judge_path
    elif dataset:
        path = "saved_models/" + dataset + str(N_to_mask)
    else:
        raise Exception("dataset must be specified")

    if dataset == "mnist":
        judge = MNISTJudge(
            N_to_mask=N_to_mask, model_dir=path, binary_rewards=binary_rewards
        )
    elif dataset == "fashion":
        judge = FashionJudge(
            N_to_mask=N_to_mask, model_dir=path, binary_rewards=binary_rewards
        )
    else:
        raise Exception("Unknown dataset in " + "dataset.txt: " + dataset)

    sample = judge.eval_data[sample_id].flatten()

    label = judge.eval_labels[sample_id]
    if lying_agent_label == label:
        lying_agent_label = (lying_agent_label + 1) % 10

    agent_lie = DebateAgent(precommit_label=lying_agent_label, agentStrength=rollouts)
    agent_truth = DebateAgent(precommit_label=label, agentStrength=rollouts)

    player_description = DebatePlayers(
        agent_truth, agent_lie, index_of_our_agent, our_name="truth", opp_name="liar"
    )

    debate = Debate(
        player_description.agents,
        judge,
        N_to_mask,
        sample,
        debug=True,
        changing_sides=changing_sides,
    )
    utility = debate.play()
    player_description.print_debate_result(utility, label)
