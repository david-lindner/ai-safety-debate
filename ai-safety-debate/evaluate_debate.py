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
    judge_path = "./mnist2000judge"
    dataset = "mnist"
    nmbr_samples = 10
    paper_eval = True
    rollouts = 10000


@ex.automain
def run(N_to_mask, judge_path, dataset, nmbr_samples, paper_eval, rollouts):
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

    if not nmbr_samples:
        nmbr_samples = len(judge.eval_data)

    truth_agent_wins = 0
    sample_count = 0

    for sample_id in range(nmbr_samples):
        sample_count += 1
        sample = judge.eval_data[sample_id].flatten()

        label = judge.eval_labels[sample_id]

        if paper_eval:
            for lying_agent_label in range(10):
                print(lying_agent_label)
                if lying_agent_label == label:
                    continue
                winner = 1
                liar_wins = 0
                for game in range(3):
                    agent1 = Agent(
                        precommit_label=lying_agent_label, agentStrength=rollouts
                    )
                    agent2 = Agent(precommit_label=label, agentStrength=rollouts)

                    debate = Debate((agent1, agent2), judge, N_to_mask, sample)
                    this_game_winner = debate.play()
                    if this_game_winner == 0:
                        liar_wins = liar_wins + 1
                    if liar_wins == 2 or (liar_wins == 0 and game == 1):
                        break

                if liar_wins >= 2:
                    winner = 0
                    break

        else:  # fast evaluation with random  lying label
            lying_agent_label = randint(0, 9)
            while label == lying_agent_label:
                lying_agent_label = randint(0, 9)

            agent1 = Agent(precommit_label=lying_agent_label, agentStrength=rollouts)
            agent2 = Agent(precommit_label=label, agentStrength=rollouts)

            debate = Debate((agent1, agent2), judge, N_to_mask, sample)
            winner = debate.play()

        if winner == 0:
            print(":(")
            print(
                "Sample {}:  Truth wins {} out of {} ({}%)".format(
                    sample_count,
                    truth_agent_wins,
                    sample_count,
                    100 * truth_agent_wins / sample_count,
                )
            )
        else:
            truth_agent_wins = truth_agent_wins + 1
            print(":)")
            print(
                "Sample {}:  Truth wins {} out of {} ({}%)".format(
                    sample_count,
                    truth_agent_wins,
                    sample_count,
                    100 * truth_agent_wins / sample_count,
                )
            )

    print(
        "Result:  Truth wins {} out of {} ({}%)".format(
            truth_agent_wins, nmbr_samples, 100 * truth_agent_wins / nmbr_samples
        )
    )
