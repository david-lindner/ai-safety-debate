import numpy as np

from judge import MNISTJudge
from agent import DebateClassifier


if __name__ == "__main__":
    judge = MNISTJudge(4)
    train_data = judge.train_data
    train_labels = judge.train_labels
    eval_data = judge.eval_data
    eval_labels = judge.eval_labels

    debate_classifier = DebateClassifier()
    for i in range(100):
        batch_start = (i * 128) % len(train_data)
        batch_end = min(batch_start + 128, len(train_data))
        batch = train_data[batch_start:batch_end]
        labels = train_labels[batch_start:batch_end]
        loss_weights = np.ones_like(labels, dtype=np.float32)
        debate_classifier.train(batch, labels, loss_weights)
    acc = debate_classifier.evaluate_accuracy(eval_data, eval_labels)
    print("Accuracy", acc)
