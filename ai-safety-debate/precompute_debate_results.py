import argparse
import numpy as np

from judge import MNISTJudge, FashionJudge
from debate import Debate
from agent import DebateAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, help="Currently only 'mnist' or 'fashion'"
    )
    parser.add_argument(
        "--N-to-mask", type=int, help="Number of features revealed as an input"
    )
    parser.add_argument("--judge-path", type=str, help="Path to load the judge from")
    parser.add_argument("--rollouts", type=int, help="Number of rollouts for MCTS")
    parser.add_argument("--outfile", type=str, help="Path to save the results to")

    args = parser.parse_args()

    if args.dataset == "mnist":
        judge = MNISTJudge(N_to_mask=args.N_to_mask, model_dir=args.judge_path)
    elif args.dataset == "fashion":
        judge = FashionJudge(N_to_mask=args.N_to_mask, model_dir=args.judge_path)
    else:
        raise Exception("Unknown dataset " + args.dataset)

    train_data = judge.train_data
    N_train = len(judge.train_labels)
    results = np.zeros((N_train, 10))

    for i in range(100):  # N_train):
        for label in range(10):
            sample = train_data[i]
            agent1 = DebateAgent(precommit_label=None, agentStrength=args.rollouts)
            agent2 = DebateAgent(precommit_label=label, agentStrength=args.rollouts)
            debate = Debate((agent1, agent2), judge, args.N_to_mask, sample.flat)
            winner = debate.play()
            results[i, label] = winner

    results.tofile(args.outfile)
