import numpy as np


class DebateState:
    def __init__(
        self, sample, initial_statements, judge, moves_left=6, starting_player=0
    ):
        # debate has to tell the state how many moves can we make
        self.sample = sample
        self.mask = np.zeros_like(sample)
        self.initial_statements = initial_statements
        self.judge = judge
        self.moves_left = moves_left
        self.currentPlayer = starting_player

    def getPossibleActions(self):
        # not selected and nonzero feature
        return np.where((self.mask == 0) & (self.sample != 0))[0]

    def takeAction(self, action):
        assert action in self.getPossibleActions()
        newState = copy.copy(self)
        newState.mask = copy.copy(self.mask)
        newState.mask[action] = 1
        newState.moves_left -= 1
        newState.currentPlayer = (self.currentPlayer + 1) % 2
        return newState

    def maximizerNode(self):
        # return +1 if the current player is a maximizer, -1 if they are a minimizer
        if self.currentPlayer == 0:
            return 1
        else:
            return -1

    def isTerminal(self):
        assert self.moves_left >= 0
        return self.moves_left == 0

    def getReward(self):
        assert self.isTerminal()
        # judge returns 0 when the first player wins, 1 when second player wins
        judge_outcome = judge.evaluateDebate(np.stack(self.mask, self.sample))
        # MCTS needs to get a high reward when first player wins and low number when second wins
        # the following line returns 0 when pl.1 wins and -1 when pl.2 wins. This is intentional.
        return judge_outcome * (-1)
        # return 666
