from judge import MNISTJudge
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Currently only 'mnist'")
    parser.add_argument(
        "--N-features", type=int, help="Number of features revealed as an input"
    )
    parser.add_argument("--train-steps", type=int, help="Number of training steps")
    parser.add_argument("--path", type=str, help="Path to save the trained judge to")
    args = parser.parse_args()

    if args.dataset == "mnist":
        judge = MNISTJudge(N_pixels=args.N_features, save_model_as=args.path)
    else:
        raise Exception("Unknown dataset " + args.dataset)

    judge.train(args.train_steps)
    print(judge.evaluate_accuracy())
