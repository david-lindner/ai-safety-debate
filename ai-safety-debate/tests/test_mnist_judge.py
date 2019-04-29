from judge import MNISTJudge


def test_mnist_judge():
    # dummy test, would make sense to write more at some point
    judge = MNISTJudge(N_to_mask=700)
    assert judge.N_to_mask == 700
