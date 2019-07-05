import argparse
import numpy as np
from judge import MNISTJudge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N-to-mask",
        type=int,
        help="Number of features revealed as an input",
        required=True,
    )
    parser.add_argument(
        "--file", type=str, required=True, help="File containing debate results"
    )
    args = parser.parse_args()

    judge = MNISTJudge(N_to_mask=args.N_to_mask)
    train_labels = judge.train_labels
    truth_wins_count, lie_loses_count = 0, 0

    debate_results = np.fromfile(args.file).reshape(-1, 10, 10)
    n_samples = len(debate_results)
    for i in range(n_samples):
        true_label = train_labels[i]
        probabilities = debate_results[i, true_label]
        if np.all(probabilities[true_label] >= probabilities):
            truth_wins_count += 1
        for wrong_label in range(10):
            if wrong_label != true_label:
                probabilities = debate_results[i, wrong_label]
                if not np.all(probabilities[wrong_label] >= probabilities):
                    lie_loses_count += 1

    print(
        "Truth won {} out of {}  ({:.2f}%)".format(
            truth_wins_count, n_samples, 100 * truth_wins_count / n_samples
        )
    )
    print(
        "Lie lost {} out of {}  ({:.2f}%)".format(
            lie_loses_count, n_samples * 9, 100 * lie_loses_count / (9 * n_samples)
        )
    )
    print(
        "Overall, debate was correct {} out of {} times ({:.2f}%)".format(
            truth_wins_count + lie_loses_count,
            n_samples * 10,
            100 * (truth_wins_count + lie_loses_count) / (10 * n_samples),
        )
    )
