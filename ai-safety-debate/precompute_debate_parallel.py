"""
TODO
"""

import sys
import os
import subprocess
import argparse
import functools
import math
import time
import numpy as np

from multiprocessing import Pool

from judge import MNISTJudge

from precompute_single_debate import ex as single_debate

# precompute_single_script = os.path.join(
    # os.path.dirname(__file__), "precompute_single_debate.py"
# )

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
    parser.add_argument("--N-jobs", type=int, help="Number of jobs", required=True)
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

    judge = MNISTJudge(
        N_to_mask=args.N_to_mask, model_dir=args.judge_path, binary_rewards=False
    )
    if args.use_test_data:
        dataset = judge.eval_data
    else:
        dataset = judge.train_data

    jobs = []
    for i in range(args.N_samples):
        print("i", args.start_sample + i, flush=True)
        if args.start_sample + i > dataset.shape[0]:  # end of dataset
            break
        results_per_label = np.zeros([10, 10])
        for label in range(10):
            if args.restricted_first:
                first_agent_label = label
                second_agent_label = None
            else:
                first_agent_label = None
                second_agent_label = label
            job = {
                "N_to_mask": args.N_to_mask,
                "use_test_data": args.use_test_data,
                "sample_id": i + args.start_sample,
                "first_agent_label": first_agent_label,
                "second_agent_label": second_agent_label,
                "dataset": "mnist",
                "judge_path": args.judge_path,
                "rollouts": args.rollouts,
                "binary_rewards": True,
                "changing_sides": False,
            }
            jobs.append(job)

    t = time.time()

    def run_job(config_updates):
        arguments = ["{}={}".format(k, v) for k, v in config_updates.items()]
        command = [sys.executable, precompute_single_script, "with"] + arguments
        print(" ".join(command))
        stderr = []
        with subprocess.Popen(
            command, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True
        ) as p:
            for line in p.stderr:
                stderr.append(line)
        returncode = p.wait()
        print("\n".join(stderr))

    t = time.time()
    # with Pool(args.N_jobs) as p:
        # p.map(run_job, jobs)
    for job in jobs:
        # run_job(job)
        single_debate.run(config_updates=job)
    print("time", time.time() - t)
