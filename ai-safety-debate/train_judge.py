from judge import MNISTJudge, FashionJudge
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Currently only 'mnist' or 'fashion'")
    parser.add_argument(
        "--N-features", type=int, help="Number of features revealed as an input"
    )
    parser.add_argument("--train-steps", type=int, help="Number of training steps")
    parser.add_argument("--path", type=str, help="Path to save the trained judge to")
    args = parser.parse_args()

    path = args.path or "saved_models/" + args.dataset + str(args.N_features)

    if args.dataset == "mnist":
        judge = MNISTJudge(N_to_mask=args.N_features, save_model_as=path)
    elif args.dataset == "fashion":
        judge = FashionJudge(N_to_mask=args.N_features, save_model_as=path)
    else:
        raise Exception("Unknown dataset " + args.dataset)

    judge.train(args.train_steps)
    print(judge.evaluate_accuracy())