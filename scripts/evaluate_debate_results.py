import jsonlines
import numpy as np

if __name__ == '__main__':
    filename = "debate_dump_mnist_4_rollouts_1000.jsonl"

    truth_wins_count = 0
    count = 0
    sample_ids = []
    with jsonlines.open(filename, "r") as reader:
        for experiment in reader:
            sample_id = int(experiment["config"]["sample_id"])
            sample_ids.append(sample_id)
            true_label = int(experiment["true_label"])
            first_agent_label = experiment["config"]["first_agent_label"]
            second_agent_label = experiment["config"]["second_agent_label"]
            probabilities = np.array(experiment["result"])
            count += 1
            if np.all(probabilities[true_label] >= probabilities):
                truth_wins_count += 1
    print("Samples {} to {}".format(min(sample_ids), max(sample_ids)))
    print("Truth wins {}/{} ({:.2f}%)".format(truth_wins_count, count, truth_wins_count/count))
