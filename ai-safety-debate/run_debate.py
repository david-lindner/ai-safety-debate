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
    index_of_true_agent = 1
    binary_rewards = False
    changing_sides = True


@ex.automain
def run(
    N_to_mask,
    sample_id,
    lying_agent_label,
    judge_path,
    dataset,
    rollouts,
    index_of_true_agent,
    binary_rewards,
    changing_sides,
):
    """
    Runs one debate game on a given sample from a specified dataset (either "mnist" or "fashion") with N_to_mask rounds. 
    The debate is modeled by MCTS with given number of rollouts.
    One player is the honest agents arguing for the correct label, the other either precommits to a lying_agent_label or debates without precommit if lying_agent_label is set to None. 

    index_of_true_agent: Either 0 or 1 whether the honest agent plays first or second.
    binary_rewards: If set to True, rewards are either 1 or -1, if set False rewards lie between -1 and 1
    changing_sides: If set to True, agents switch sides after each round, if set to False, the first agents reveales N_to_mask/2 features followed by N_to_mask/2 features of the second agent
    """
    # Parse parameters
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

    # Get debate data and run the debate
    sample = judge.eval_data[sample_id].flatten()

    label = judge.eval_labels[sample_id]

    if lying_agent_label == label:
        raise Exception("Lying label cannot be the same as the correct one")

    agent_lie = DebateAgent(precommit_label=lying_agent_label, agentStrength=rollouts)
    agent_truth = DebateAgent(precommit_label=label, agentStrength=rollouts)
    assert index_of_true_agent in [0, 1]

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
