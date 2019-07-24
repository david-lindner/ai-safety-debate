import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge
from agent import DebateAgent, DebatePlayers
from debate import Debate

ex = Experiment("tabular_mdp_experiment")
ex.observers.append(FileStorageObserver.create("precompute"))


@ex.config
def cfg():
    N_to_mask = 4
    use_test_data = False
    sample_id = np.random.randint(100)
    first_agent_label = None
    second_agent_label = None
    dataset = "mnist"
    judge_path = None
    rollouts = 100
    binary_rewards = False
    changing_sides = True


@ex.automain
def run(
    _run,
    N_to_mask,
    use_test_data,
    sample_id,
    first_agent_label,
    second_agent_label,
    dataset,
    judge_path,
    rollouts,
    binary_rewards,
    changing_sides,
):
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
    _run.log_scalar("true_label", label)

    first_agent = DebateAgent(precommit_label=first_agent_label, agentStrength=rollouts)
    second_agent = DebateAgent(
        precommit_label=second_agent_label, agentStrength=rollouts
    )

    debate = Debate(
        [first_agent, second_agent],
        judge,
        N_to_mask,
        sample,
        debug=False,
        changing_sides=changing_sides,
    )
    print("Starting debate")
    t = time.time()
    probabilities = debate.play(full_report=True)
    print("Debate done in {} seconds".format(time.time() - t))
    return probabilities
