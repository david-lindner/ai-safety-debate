import numpy as np
import copy


class DebateState:
    def __init__(
        self,
        sample,
        initial_statements,
        judge,
        moves_left=6,
        starting_player=0,
        player_order=None,
    ):
        # debate has to tell the state how many moves can we make
        self.sample = sample
        self.mask = np.stack((np.zeros_like(sample), np.zeros_like(sample)))
        self.initial_statements = initial_statements
        self.judge = judge
        self.moves_left = moves_left
        if player_order == None:
            raise Exception("Player order not specified")

        # player order is an array determinating which player should play each round
        # starting_player is not used anymore
        self.currentPlayer = player_order[0]
        self.player_order = player_order

    def getPossibleActions(self):
        # not selected and nonzero feature
        return np.where((self.mask.sum(axis=0) == 0) & (self.sample != 0))[0]

    def takeAction(self, action):
        assert action in self.getPossibleActions()
        newState = copy.copy(self)
        newState.mask = np.copy(self.mask)
        newState.mask[newState.currentPlayer, action] = 1
        newState.moves_left -= 1
        moves_past = len(newState.player_order) - newState.moves_left
        if (
            newState.player_order[moves_past] != 0
            and newState.player_order[moves_past] != 1
        ):
            raise Exception("Player order elements can be only either 0 or 1 ")
        newState.currentPlayer = newState.player_order[-newState.moves_left]
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
        mask = self.mask.sum(axis=0)
        utility = self.judge.evaluate_debate(
            np.stack((mask, self.sample * mask), axis=1), self.initial_statements
        )
        assert -1 <= utility & utility <= 1
        return utility
