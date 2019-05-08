"""
Run the debate evaluation on a dataset. Many features included.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from os import remove

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import DebateAgent

ex = Experiment("mnist_debate_eval")
ex.observers.append(FileStorageObserver.create("amplification_experiments"))
fashion_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@ex.config
def cfg():
    N_to_mask = 4
    judge_path = None
    dataset = "mnist"
    nmbr_samples = 10
    start_at_sample = 0
    eval_unrestricted = False
    rollouts = 10
    index_of_truth_agent = 0
    changing_sides = True
    compute_confusion_matrix = False


def evaluate_sample_restricted(N_to_mask, sample, label, judge, rollouts, index_of_truth_agent, changing_sides, seeds=3, confusion_matrix_counter=None):
    """
    Reproduce the experiment from AI safety via debate paper, that is with 3 game seeds.
    If confusion_matrix_counter is specified, than update the counter so that confusion_matrix_counter[true_label][lying_label] specifies the number of times 'lying_label' won over 'true_label'.
    Use only the first game seed to update the counter.
    Different number of seeds actualy won't work.

    """
    truth_won = True
    for lying_agent_label in range(10):
        if lying_agent_label == label:
            continue
        print(lying_agent_label, end="-", flush=True)
        liar_win_count = 0
        for game_number in range(seeds):
            agent_lie = DebateAgent(
                precommit_label=lying_agent_label, agentStrength=rollouts
            )
            agent_truth = DebateAgent(
                precommit_label=label, agentStrength=rollouts
            )
            if index_of_truth_agent == 0:
                debate = Debate(
                    (agent_truth, agent_lie),
                    judge,
                    N_to_mask,
                    sample,
                    debug=False,
                    changing_sides=changing_sides,
                )
                this_game_utility = debate.play()
                if this_game_utility == -1:  # second agent won (lying)
                    liar_win_count += 1
                    if confusion_matrix_counter is not None and game_number == 0:
                        confusion_matrix_counter[label][lying_agent_label] += 1
            else:
                debate = Debate(
                    (agent_lie, agent_truth),
                    judge,
                    N_to_mask,
                    sample,
                    debug=False,
                    changing_sides=changing_sides,
                )
                this_game_utility = debate.play()
                if this_game_utility == 1:  # first agent won (lying)
                    liar_win_count += 1
                    if confusion_matrix_counter is not None and game_number == 0:
                        confusion_matrix_counter[label][lying_agent_label] += 1
            # If liar's already won majority of games for the label and we're not computing the confusion matrix, we can end the evaluation on this sample
            if liar_win_count == 2 or (
                liar_win_count == 0 and game_number == 1
            ):
                if confusion_matrix_counter is None:
                    break
        if liar_win_count >= 2:
            truth_won = False
            if confusion_matrix_counter is None:
                break
    return truth_won


def evaluate_sample_unrestricted(N_to_mask, sample, label, judge, rollouts, index_of_truth_agent, changing_sides):
    """
    Evaluate unrestricted debate (without precommit)
    """
    truth_won = True
    agent_lie = DebateAgent(precommit_label=None, agentStrength=rollouts)
    agent_truth = DebateAgent(precommit_label=label, agentStrength=rollouts)

    if index_of_truth_agent == 0:
        debate = Debate(
            (agent_truth, agent_lie),
            judge,
            N_to_mask,
            sample,
            debug=False,
            changing_sides=changing_sides,
        )
        this_game_utility = debate.play()
        if this_game_utility == -1:
            truth_won = False
    else:
        debate = Debate(
            (agent_lie, agent_truth),
            judge,
            N_to_mask,
            sample,
            debug=False,
            changing_sides=changing_sides,
        )
        this_game_utility = debate.play()
        if this_game_utility == 1:
            truth_won = False
    return truth_won

@ex.automain
def run(
    N_to_mask,
    judge_path,
    dataset,
    nmbr_samples,
    start_at_sample,
    eval_unrestricted,
    rollouts,
    index_of_truth_agent,
    changing_sides,
    compute_confusion_matrix,
):

    """
    Evaluates debate on a given number of samples of a given dataset ("mnist", "fashion").
    Each debate has N_to_mask rounds.

    The debate is either modeled with precommit or unrestricted giver the eval_unrestricted parameter.
    The precommited debate is evaluated by the way described in the "AI safety via debate" paper,
    the unrestricted debate is played once for each sample.

    index_of_truth_agent: Either 0 or 1 whether the honest agent plays first or second.
    changing_sides: If set to True, agents switch sides after each move, if set to False, the first agents reveales N_to_mask/2 features followed by N_to_mask/2 features of the second agent
    compute_confusion_matrix: If True, compute confusion matrix as in figure 3 in the AISvD paper. Only for restricted debate.

    """
    # Parse parameters
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

    print("Parameters")
    print("--------")
    print("N_to_mask:", N_to_mask)
    print("judge_path:", judge_path)
    print("dataset:", dataset)
    print("nmbr_samples:", nmbr_samples)
    print("start_at_sample:", start_at_sample)
    print("eval_unrestricted:", eval_unrestricted)
    print("rollouts:", rollouts)
    print("index_of_truth_agent:", index_of_truth_agent)
    print("changing_sides:", changing_sides)
    print("compute_confusion_matrix:", compute_confusion_matrix)
    print("--------")
    judge_accuracy = judge.evaluate_accuracy()
    print("Judge accuracy:", judge_accuracy)
    print("--------")

    # Prepare for confusion
    # confusion_matrix_counter[true_label][lying_label] specifies the number of times 'lying_label' won over 'true_label'
    # labels_frequency counts the occurancies of each label
    if compute_confusion_matrix:
        if eval_unrestricted:
            raise Exception("Consusion matrix can be computed only while evaluating restricted debate")
        confusion_matrix_counter = np.array([[0 for i in range(10)] for i in range(10)], dtype='f')
        labels_frequency = [0 for i in range(10)]
    else:
        confusion_matrix_counter = None

    # Run debate for each sample
    overall_truth_win_count = 0
    for sample_id in range(start_at_sample, start_at_sample + nmbr_samples):
        sample_start_time = time.time()
        sample = judge.eval_data[sample_id].flatten()
        label = judge.eval_labels[sample_id]
        truth_won = True

        # Reproduce the experiment from AI safety via debate paper
        if not eval_unrestricted:
            truth_won = evaluate_sample_restricted(N_to_mask, sample, label, judge, rollouts, index_of_truth_agent, changing_sides, seeds=3, confusion_matrix_counter=confusion_matrix_counter)
            if compute_confusion_matrix:
                labels_frequency[label] += 1

        # Evaluate unrestricted debate (without precommit)
        else:
            truth_won = evaluate_sample_unrestricted(N_to_mask, sample, label, judge, rollouts, index_of_truth_agent, changing_sides)

        print("\t Sample {}".format(sample_id + 1), end=" ", flush=True)
        if truth_won:
            overall_truth_win_count += 1
            print("Winner: Truth.", end=" ", flush=True)
        else:
            print("Winner: Liar.", end=" ", flush=True)
        print(
            "Truth winrate: {} out of {} ({}%)".format(
                overall_truth_win_count,
                sample_id - start_at_sample + 1,
                100 * overall_truth_win_count / (sample_id - start_at_sample + 1),
            ),
            flush=True,
        )
        print("\t  Sample time: {}".format(time.time() - sample_start_time))

    print(
        "Overall truth winrate: {} out of {} ({}%)".format(
            overall_truth_win_count,
            nmbr_samples,
            100 * overall_truth_win_count / nmbr_samples,
        ),
        flush=True,
    )

    # Computes, prints and saves the confusion matrix into the experiment folder
    if compute_confusion_matrix:
        print("Unnormalized confusion matrix:")
        print(confusion_matrix_counter)

        print("Frequency of labels:")
        print(labels_frequency)

        # normalize according to frequencies of the labels and multiply by 100 
        normalized_confusion_matrix = confusion_matrix_counter
        for label in range(10):
            if labels_frequency[label] != 0:
                normalized_confusion_matrix[label] = confusion_matrix_counter[label] * 1/labels_frequency[label]*100
        print("Normalized confusion matrix:")
        print(normalized_confusion_matrix)

        # adjust figure size
        if dataset == "mnist":
            plt.figure(figsize=(6,6))
        elif dataset == "fashion":
            plt.figure(figsize=(8,7))
        else:
            plt.figure(figsize=(10,5))
        plt.matshow(normalized_confusion_matrix, cmap=plt.cm.jet, fignum=1)

        cbar = plt.colorbar()
        cbar.set_label('Liar\'s wins percentage')

        # TODO liar label at the top
        plt.xlabel('Liar label')
        plt.ylabel('True label')


        if dataset == "mnist":
            plt.xticks(np.arange(10))
            plt.yticks(np.arange(10))
        elif dataset == "fashion":
            plt.xticks(np.arange(10), fashion_labels, rotation='vertical')
            plt.yticks(np.arange(10), fashion_labels)

        plt.savefig('confusion_matrix.png')
        ex.add_artifact('confusion_matrix.png')
        remove('confusion_matrix.png')
        # uncomment to plot the matrix
        # plt.show()
