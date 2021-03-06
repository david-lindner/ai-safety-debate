"""
Minor script used for debugging.

The purpose is to ensure the judge's 'predictor' and 'estimator' yield the same
prediction accuracy (because they are supposed to contain the same model).
"""

from judge import MNISTJudge, FashionJudge

if __name__ == "__main__":
    dataset = "mnist"
    judge_path = "judge_mnist_4"
    N_to_mask = 4

    if judge_path:
        path = judge_path
    elif dataset:
        path = "saved_models/" + dataset + str(N_to_mask)
    else:
        raise Exception("Either judge_path or dataset needs to be specified")

    if dataset == "mnist":
        judge = MNISTJudge(N_to_mask=N_to_mask, model_dir=path)
    elif dataset == "fashion":
        judge = FashionJudge(N_to_mask=N_to_mask, model_dir=path)
    else:
        raise Exception("Unknown dataset in " + "dataset.txt: " + dataset)

    print("accuracy estimator", judge.evaluate_accuracy())
    print("accuracy predictor", judge.evaluate_accuracy_using_predictor())
