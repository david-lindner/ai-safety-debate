from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import DebateAgent

ex = Experiment("mnist_debate_eval")
ex.observers.append(FileStorageObserver.create("amplification_experiments"))


@ex.config
def cfg():
    N_to_mask = 4
    judge_path = None
    dataset = "mnist"
    nmbr_samples = 100
    rollouts = 10
    eval_unrestricted = False
    index_of_truth_agent = 0
    changing_sides = True


@ex.automain
def run(
    N_to_mask,
    judge_path,
    dataset,
    nmbr_samples,
    eval_unrestricted,
    rollouts,
    index_of_truth_agent,
    changing_sides,
):
    # parse parameters
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

    # run debate for each sample
    overall_truth_win_count = 0
    for sample_id in range(nmbr_samples):
        sample = judge.eval_data[sample_id].flatten()
        label = judge.eval_labels[sample_id]
        truth_won = True

        # reproduce the experiment from AI safety via debate paper
        if not eval_unrestricted:
            for lying_agent_label in range(10):
                if lying_agent_label == label:
                    continue
                print(lying_agent_label, end="-", flush=True)
                liar_win_count = 0
                for game_number in range(3):
                    agent_lie = DebateAgent(
                        precommit_label=lying_agent_label, agentStrength=rollouts
                    )
                    agent_truth = DebateAgent(
                        precommit_label=label, agentStrength=rollouts
                    )
                    # TODO this will do weird stuff if the utilities are non-binary. Works if they are either 1 or -1.
                    if index_of_truth_agent == 0:
                        debate = Debate(
                            (agent_truth, agent_lie),
                            judge,
                            N_to_mask,
                            sample,
                            debug=False,
                            simultaneous=simultaneous,
                        )
                        this_game_utility = debate.play()
                        if this_game_utility == -1:  # second agent won (lying)
                            liar_win_count += 1
                    else:
                        debate = Debate(
                            (agent_lie, agent_truth),
                            judge,
                            N_to_mask,
                            sample,
                            debug=False,
                            simultaneous=simultaneous,
                        )
                        this_game_utility = debate.play()
                        if this_game_utility == 1:  # first agent won
                            liar_win_count += 1
                    if liar_win_count == 2 or (
                        liar_win_count == 0 and game_number == 1
                    ):
                        break
                if liar_win_count >= 2:
                    truth_won = False
                    break

        # evaluate unrestricted
        else:
            agent_lie = DebateAgent(precommit_label=None, agentStrength=rollouts)
            agent_truth = DebateAgent(precommit_label=label, agentStrength=rollouts)

            if index_of_truth_agent == 0:
                debate = Debate(
                    (agent_truth, agent_lie), judge, N_to_mask, sample, debug=False, changing_sides=changing_sides,
                )
                this_game_utility = debate.play()
                if this_game_utility == -1:
                    truth_won = False
            else:
                debate = Debate(
                    (agent_lie, agent_truth), judge, N_to_mask, sample, debug=False, changing_sides=changing_sides,
                )
                this_game_utility = debate.play()
                if this_game_utility == 1:
                    truth_won = False
        if truth_won:
            overall_truth_win_count += 1
            print("\t Winner: Truth.", end=" ")
        else:
            print("\t Winner: Liar.", end=" ")
        print(
            "Truth winrate: {} out of {} ({}%)".format(
                overall_truth_win_count,
                sample_id + 1,
                100 * overall_truth_win_count / (sample_id + 1),
            )
        )

    print(
        "Overall truth winrate: {} out of {} ({}%)".format(
            overall_truth_win_count,
            nmbr_samples,
            100 * overall_truth_win_count / nmbr_samples,
        )
    )
