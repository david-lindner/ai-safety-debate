"""
Trains and saves a judge to be used for any of the other scripts.
Pretrained judges can be founed in our Google Drive folder.
"""

from judge import MNISTJudge, FashionJudge
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, help="Currently only 'mnist' or 'fashion'"
    )
    parser.add_argument(
        "--N-to-mask", type=int, help="Number of features revealed as an input"
    )
    parser.add_argument(
        "--train-steps",type=int, 
        help="Number of training steps. If more than 1, apply to corresponding index of n-zero"
    )
    parser.add_argument(
        "--n-zero", nargs="*", type=int, help="Number of 0-pixels to sample"
    )
    parser.add_argument(
        "--path", type=str, help="Path to save the trained judge to (and restore from)"
    )
    args = parser.parse_args()

    path = args.path or "saved_models/" + args.dataset + str(args.N_to_mask)

    if args.dataset == "mnist":
        judge = MNISTJudge(N_to_mask=args.N_to_mask, model_dir=path)
    elif args.dataset == "fashion":
        judge = FashionJudge(N_to_mask=args.N_to_mask, model_dir=path)
    else:
        raise Exception("Unknown dataset " + args.dataset)

    t = time.time()
    judge.train(args.train_steps, args.n_zero if len(args.n_zero) > 1 else args.n_zero[0])
    print('Time', time.time() - t)
    print('Accuracy', judge.evaluate_accuracy(args.n_zero))
