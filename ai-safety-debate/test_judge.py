from judge import MNISTJudge

if __name__ == "__main__":
    judge = MNISTJudge(N_pixels=4)
    judge.train(1000)
    judge.evaluate()
