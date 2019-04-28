from judge import MNISTJudge

def test_mnist_judge():
    # dummy test, would make sense to write more at some point
    judge = MNISTJudge(N_pixels=700)
    assert judge.N_pixels == 700
