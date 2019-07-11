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
    truth_wins_count, restricted_wins_count = 0, 0

    debate_results = np.fromfile(args.file).reshape(-1, 10, 10)
    n_samples = debate_results.shape[0]
    for i in range(n_samples):
        true_label = train_labels[i]
        for label in range(10):
            judge_probabilities = debate_results[i, label]
            if np.all(judge_probabilities[label] >= judge_probabilities):
                restricted_wins_count += 1
                if label == true_label:
                    truth_wins_count += 1

    print("\n-------------------------------")
    print("Evaluation of precomputed debate results in:", args.file)
    print("-------------------------------")

    print(
        "The restricted agent won {} out of {} debates ({:.2f}%)".format(
            restricted_wins_count,
            n_samples * 10,
            100 * restricted_wins_count / (10 * n_samples),
        )
    )
    print(
        "Out of these {} wins, it won with the true label {} times".format(
            restricted_wins_count, truth_wins_count
        )
    )
    print(
        "Hence the debate 'precision' is {:.2f}%".format(
            100 * truth_wins_count / restricted_wins_count
        )
    )
