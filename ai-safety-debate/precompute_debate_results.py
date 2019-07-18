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


def get_debate_results(
    start_point,
    use_test_data,
    batch_size,
    N_samples,
    N_to_mask,
    judge_path,
    restricted_first,
):
    # MNISTJudge has to be imported here, because otherwise tensorflow does not
    # work together with multiprocessing
    from judge import MNISTJudge

    judge = MNISTJudge(N_to_mask=N_to_mask, model_dir=judge_path, binary_rewards=False)
    if use_test_data:
        dataset = judge.eval_data
    else:
        dataset = judge.train_data

    result_list = []
    for i in range(batch_size):
        print("i", start_point + i, flush=True)
        t = time.time()
        if start_point + i > dataset.shape[0]:  # end of dataset
            break
        results_per_label = np.zeros([10, 10])
        for label in range(10):
            # print("label", label)
            sample = dataset[start_point + i]
            unrestricted_agent = DebateAgent(
                precommit_label=None, agentStrength=args.rollouts
            )
            restricted_agent = DebateAgent(
                precommit_label=label, agentStrength=args.rollouts
            )
            if restricted_first:
                agent1, agent2 = restricted_agent, unrestricted_agent
            else:
                agent1, agent2 = unrestricted_agent, restricted_agent
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
        "--use-test-data",
        help="If set to true, the test set will be used instead of the train set.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--start-sample",
        type=int,
        help="Start at a specific training example (to split up the precomputation across jobs).",
        default=0,
    )
    parser.add_argument(
        "--N-samples",
        type=int,
        help="Number of samples to precompute debate results for.",
        default=60000,
    )
    parser.add_argument("--restricted-first", action="store_true")

    args = parser.parse_args()

    results = np.zeros((args.N_samples, 10))
    batch_size = math.ceil(args.N_samples / args.N_threads)
    start_points = [args.start_sample + i * batch_size for i in range(args.N_threads)]

    get_debate_results_partial = functools.partial(
        get_debate_results,
        use_test_data=args.use_test_data,
        batch_size=batch_size,
        N_samples=args.N_samples,
        N_to_mask=args.N_to_mask,
        judge_path=args.judge_path,
        restricted_first=args.restricted_first,
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
