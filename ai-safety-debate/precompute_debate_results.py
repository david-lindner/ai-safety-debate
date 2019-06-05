"""
Calculates results of non-precommited debates for all MNIST evaluation images
and stores writes them as a serialized numpy array to a file. Provides the
option to paralellize this process using `multiprocessing`. Currently only
works for MNIST.
"""

import argparse
import functools
import math
import time
import numpy as np

from multiprocessing import Pool

from debate import Debate
from agent import DebateAgent


def get_debate_results(start_point, batch_size, N_train, N_to_mask, judge_path):
    # MNISTJudge has to be imported here, because otherwise tensorflow does not
    # work together with multiprocessing
    from judge import MNISTJudge

    judge = MNISTJudge(N_to_mask=N_to_mask, model_dir=judge_path, binary_rewards=False)
    train_data = judge.train_data

    result_list = []
    for i in range(batch_size):
        print("i", start_point + i, flush=True)
        t = time.time()
        if start_point + i > N_train:  # end of dataset
            break
        results_per_label = np.zeros([10, 10])
        for label in range(10):
            # print("label", label)
            sample = train_data[start_point + i]
            agent1 = DebateAgent(precommit_label=None, agentStrength=args.rollouts)
            agent2 = DebateAgent(precommit_label=label, agentStrength=args.rollouts)
            debate = Debate((agent1, agent2), judge, N_to_mask, sample.flat)
            probabilities = debate.play(full_report=True)
            results_per_label[label] = probabilities
        result_list.append(results_per_label)
        print("time", time.time() - t)
    return result_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N-to-mask",
        type=int,
        help="Number of features revealed as an input",
        required=True,
    )
    parser.add_argument(
        "--judge-path", type=str, help="Path to load the judge from", required=True
    )
    parser.add_argument(
        "--rollouts", type=int, help="Number of rollouts for MCTS", required=True
    )
    parser.add_argument(
        "--outfile", type=str, help="Path to save the results to", required=True
    )
    parser.add_argument(
        "--N-threads", type=int, help="Number of threads", required=True
    )
    parser.add_argument(
        "--N-train",
        type=int,
        help="Number of training labels to precompute debate results for."
        "Can be used to test this for a small set of images.",
        default=60000,
    )

    args = parser.parse_args()

    results = np.zeros((args.N_train, 10))
    batch_size = math.ceil(args.N_train / args.N_threads)
    start_points = [i * batch_size for i in range(args.N_threads)]

    get_debate_results_partial = functools.partial(
        get_debate_results,
        batch_size=batch_size,
        N_train=args.N_train,
        N_to_mask=args.N_to_mask,
        judge_path=args.judge_path,
    )

    # this should give the same result as the code below, not using multiprocessing
    # debate_results = []
    # for sp in start_points:
    #     debate_results.append(get_debate_results_partial(sp))

    t = time.time()

    with Pool(args.N_threads) as pool:
        debate_results = pool.map(get_debate_results_partial, start_points)

    print("time", time.time() - t)

    results = np.array(debate_results)
    results.tofile(args.outfile)
