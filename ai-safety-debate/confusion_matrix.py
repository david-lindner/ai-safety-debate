"""
Computes, saves and shows the confusion matrix as in Figure 3 in AISvD paper
normalized_confusion_matrix[true_label][liar_label] ... the percentage of liar's wins while arguing for 'liar_label' and the correct label being 'true_label'
"""


import numpy as np
import matplotlib.pyplot as plt
from os import remove

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import DebateAgent, DebatePlayers

ex = Experiment("confusion matrix")
ex.observers.append(FileStorageObserver.create("experiments"))

fashion_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@ex.config
def cfg():
    N_to_mask = 4
    nmbr_of_samples = 20
    judge_path = None
    dataset = "mnist"
    rollouts = 1000
    index_of_true_agent = 0
    binary_rewards = True
    changing_sides = True


@ex.automain
def run(
    N_to_mask,
    nmbr_of_samples,
    judge_path,
    dataset,
    rollouts,
    index_of_true_agent,
    binary_rewards,
    changing_sides,
):
    """
    Parameters as always.
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

    index_of_lying_agent = (index_of_true_agent + 1)%2
    
    # confusion_matrix_counter[label][lying_agent_label] = number of times the lying agent won 
    # by arguing for the 'lying_agent_label' while the true label being "label"
    confusion_matrix_counter = np.array([[0 for i in range(10)] for i in range(10)], dtype='f')

    # counts the types of labels of samples
    labels_frequency = [0 for i in range(10)]

    # play a debate game for each sample and lying label
    for sample_id in range(nmbr_of_samples):

        print("Sample", sample_id)
        sample = judge.eval_data[sample_id].flatten()

        label = judge.eval_labels[sample_id]

        labels_frequency[label] += 1
        
        for lying_agent_label in range(10):

            if lying_agent_label == label:
                continue

            agent_lie = DebateAgent(precommit_label=lying_agent_label, agentStrength=rollouts)
            agent_truth = DebateAgent(precommit_label=label, agentStrength=rollouts)
            assert index_of_true_agent in [0, 1]

            player_description = DebatePlayers(
                agent_lie, agent_truth, index_of_lying_agent, our_name="liar", opp_name="truth"
            )

            debate = Debate(
                player_description.agents,
                judge,
                N_to_mask,
                sample,
                debug=False,
                changing_sides=changing_sides,
            )
            utility = debate.play()

            liar_won = player_description.increment_our_wincount(utility)

            confusion_matrix_counter[label][lying_agent_label] += liar_won

    print("Unnormalized confusion matrix")
    print(confusion_matrix_counter)

    print("Frequency of labels")
    print(labels_frequency)

    # normalize according to frequencies of the labels and multiply by 100 
    normalized_confusion_matrix = confusion_matrix_counter
    for label in range(10):
        if labels_frequency[label] != 0:
            normalized_confusion_matrix[label] = confusion_matrix_counter[label] * 1/labels_frequency[label]*100
    print("Normalized confusion matrix")
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
    cbar.set_label('Liar win percentage')

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

    plt.show()
