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
    classifier_path = None


@ex.automain
def run(
    N_to_mask, judge_path, dataset, rollouts, N_epochs, batch_size, classifier_path
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

    train_data = judge.train_data
    N_train = len(judge.train_labels)
    eval_data = judge.eval_data
    eval_labels = judge.eval_labels
    debate_classifier = DebateClassifier(model_dir=classifier_path)

    batch_samples = []
    batch_labels = []
    batch_weights = []

    for epoch in range(N_epochs):
        for i in range(N_train):
            sample = train_data[i]
            probs = next(debate_classifier.predict(sample))["probabilities"]
            label = np.random.choice(range(len(probs)), p=probs)
            probs[label] = 0
            probs /= probs.sum()
            label2 = np.random.choice(range(len(probs)), p=probs)
            print("i", i, "label", label)
            print("i", i, "label2", label2)

            agent1 = DebateAgent(precommit_label=label, agentStrength=rollouts)
            agent2 = DebateAgent(precommit_label=label2, agentStrength=rollouts)
            debate = Debate((agent1, agent2), judge, N_to_mask, sample.flat)
            winner = debate.play()
            winner = 0 if label == judge.train_labels[i] else 1
            weight = 1 if winner == 0 else 0
            print("weight", weight)
            batch_samples.append(sample)
            batch_labels.append(label)
            batch_weights.append(weight)

            if (i + 1) % batch_size == 0 or i == N_train - 1:
                debate_classifier.train(
                    np.array(batch_samples),
                    np.array(batch_labels),
                    np.array(batch_weights),
                )
                acc = debate_classifier.evaluate_accuracy(eval_data, eval_labels)
                print("Updated model")
                print("Evaluation accuracy", acc)

    acc = debate_classifier.evaluate_accuracy(eval_data, eval_labels)
    print("Accuracy", acc)
