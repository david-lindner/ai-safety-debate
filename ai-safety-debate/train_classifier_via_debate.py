"""
Train a MNIST classifier from a sparse judge combined with a debate.
"""
import time
import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import DebateAgent, DebateClassifier

ex = Experiment("train_classifier_via_debate")
ex.observers.append(FileStorageObserver.create("experiments"))


@ex.config
def cfg():
    N_to_mask = 4
    judge_path = None
    dataset = None
    rollouts = 1000
    N_epochs = 1
    batch_size = 128
    learning_rate = 1e-4
    learning_rate_decay = False
    classifier_path = None
    cheat_debate = False
    only_update_for_wins = True
    precomputed_debate_results_restricted_first_path = None
    precomputed_debate_results_restricted_second_path = None
    shuffle_batches = True
    use_dropout = True
    importance_sampling_weights = False
    importance_sampling_cap = None


@ex.automain
def run(
    N_to_mask,
    judge_path,
    dataset,
    rollouts,
    N_epochs,
    batch_size,
    learning_rate,
    learning_rate_decay,
    classifier_path,
    cheat_debate,
    only_update_for_wins,
    precomputed_debate_results_restricted_first_path,
    precomputed_debate_results_restricted_second_path,
    shuffle_batches,
    use_dropout,
    importance_sampling_weights,
    importance_sampling_cap,
):
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

    judge_accuracy = judge.evaluate_accuracy()
    print("Judge accuracy:", judge_accuracy)

    if precomputed_debate_results_restricted_first_path is not None:
        assert precomputed_debate_results_restricted_second_path is not None
        if cheat_debate:
            raise Exception(
                "cheat_debate should not be enabled when training "
                "from precomputed debate results"
            )
        debate_results_restricted_first = np.fromfile(
            precomputed_debate_results_restricted_first_path
        ).reshape(-1, 10, 10)
        debate_results_restricted_second = np.fromfile(
            precomputed_debate_results_restricted_second_path
        ).reshape(-1, 10, 10)
        print(
            "Loaded debate results from {} and {}".format(
                precomputed_debate_results_restricted_first_path,
                precomputed_debate_results_restricted_second_path,
            )
        )
        print("These will be used for training instead of re-running the debates.")
    else:
        debate_results_restricted_first, debate_results_restricted_second = None, None

    train_data = judge.train_data
    N_train = len(judge.train_labels)
    eval_data = judge.eval_data
    eval_labels = judge.eval_labels

    debate_classifier = DebateClassifier(
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        model_dir=classifier_path,
        use_dropout=use_dropout,
    )

    batch_samples = []
    batch_labels = []
    batch_weights = []

    t = time.time()

    for epoch in range(N_epochs):
        for i in range(N_train):
            # print(i, flush=True)
            sample = train_data[i]
            probs = next(debate_classifier.predict(sample))["probabilities"]
            label = np.random.choice(range(len(probs)), p=probs)
            # label = judge.train_labels[i]
            probs[label] = 0
            probs /= probs.sum()
            # print("i", i, "label", label)
            # print("i", i, "label2", label2)
            restricted_first = np.random.random() < 0.5

            if cheat_debate:
                # simulate a perfectly accurate debate
                utility = -1 if label == judge.train_labels[i] else 0
            elif debate_results_restricted_first is not None:
                assert debate_results_restricted_second is not None
                if restricted_first:
                    debate_results = debate_results_restricted_first
                else:
                    debate_results = debate_results_restricted_second
                # use precomputed results
                judge_probabilities = debate_results[i, label]
                if np.all(judge_probabilities[label] >= judge_probabilities):
                    utility = -1
                else:
                    utility = 0
            else:
                # run non-precommited debate
                agent_unrestricted = DebateAgent(
                    precommit_label=None, agentStrength=rollouts
                )
                agent_restricted = DebateAgent(
                    precommit_label=label, agentStrength=rollouts
                )
                if restricted_first:
                    agent1, agent2 = agent_restricted, agent_unrestricted
                else:
                    agent1, agent2 = agent_unrestricted, agent_restricted
                debate = Debate((agent1, agent2), judge, N_to_mask, sample.flat)
                utility = debate.play()

            if only_update_for_wins:
                weight = 1 if utility == -1 else 0
            else:
                weight = 1 if utility == -1 else -1

            if importance_sampling_weights:
                importance_sampling_factor = 1 / probs[label]
                if (
                    importance_sampling_cap is not None
                    and importance_sampling_factor < importance_sampling_cap
                ):
                    importance_sampling_factor = importance_sampling_cap
                weight *= importance_sampling_factor

            # print("weight", weight)
            batch_samples.append(sample)
            batch_labels.append(label)
            batch_weights.append(weight)

            if (i + 1) % batch_size == 0 or i == N_train - 1:
                # update debate classifier
                print("i", i, flush=True)
                print("batch_weights", batch_weights, flush=True)
                debate_classifier.train(
                    np.array(batch_samples),
                    np.array(batch_labels),
                    np.array(batch_weights),
                    shuffle=shuffle_batches,
                )
                acc = debate_classifier.evaluate_accuracy(eval_data, eval_labels)
                print("Updated debate_classifier", flush=True)
                print("Evaluation accuracy", acc, flush=True)
                t2 = time.time()
                print("Batch time ", t2 - t)
                t = t2
                batch_samples = []
                batch_labels = []
                batch_weights = []

    acc = debate_classifier.evaluate_accuracy(eval_data, eval_labels)
    print("Accuracy", acc, flush=True)
