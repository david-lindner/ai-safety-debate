import argparse
import functools
import math
import numpy as np

from multiprocessing import Pool

from debate import Debate
from agent import DebateAgent


def get_debate_results(start_point, batch_size, N_train, N_to_mask, judge_path):
    from judge import MNISTJudge

    judge = MNISTJudge(N_to_mask=N_to_mask, model_dir=None)
    train_data = judge.train_data

    result_list = []
    for i in range(batch_size):
        print("i", start_point + i, flush=True)
        if start_point + i > N_train:
            break
        results_per_label = np.zeros(10)
        for label in range(10):
            # print("label", label)
            sample = train_data[start_point + i]
            agent1 = DebateAgent(precommit_label=None, agentStrength=args.rollouts)
            agent2 = DebateAgent(precommit_label=label, agentStrength=args.rollouts)
            debate = Debate((agent1, agent2), judge, N_to_mask, sample.flat)
            winner = debate.play()
            results_per_label[label] = winner
        result_list.append(results_per_label)
    return result_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N-to-mask", type=int, help="Number of features revealed as an input"
    )
    parser.add_argument("--judge-path", type=str, help="Path to load the judge from")
    parser.add_argument("--rollouts", type=int, help="Number of rollouts for MCTS")
    parser.add_argument("--outfile", type=str, help="Path to save the results to")
    parser.add_argument("--N-threads", type=int, help="Number of threads")

    args = parser.parse_args()

    N_train = 40
    results = np.zeros((N_train, 10))

    batch_size = math.ceil(N_train / args.N_threads)
    start_points = [i * batch_size for i in range(args.N_threads)]

    get_debate_results_partial = functools.partial(
        get_debate_results,
        batch_size=batch_size,
        N_train=N_train,
        N_to_mask=args.N_to_mask,
        judge_path=args.judge_path,
    )

    # debate_results = []
    # for sp in start_points:
    #     debate_results.append(get_debate_results_partial(sp))

    with Pool(args.N_threads) as pool:
        debate_results = pool.map(get_debate_results_partial, start_points)

    results = np.array(debate_results)
    results.tofile(args.outfile)
