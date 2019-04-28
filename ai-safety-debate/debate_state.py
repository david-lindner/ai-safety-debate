# from __future__ import division

from copy import deepcopy

# from mcts import mcts
from functools import reduce
import operator


class DebateState:
    def __init__(self):
        # TODO
        # self.data = ...
        self.currentPlayer = 1

    def getPossibleActions(self):
        # TODO
        possibleActions = []
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        # TODO
        newState.currentPlayer = self.currentPlayer * -1
        return newState

    def maximizerNode(self):
        # return +1 if the current player is a maximizer, -1 if they are a minimizer
        return self.currentPlayer

    def isTerminal(self):
        # TODO
        return False

    def getReward(self):
        # TODO plug in the judge here
        assert self.isTerminal()
        return 666


class Action:
    # TODO this is the old implementation
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.x == other.x
            and self.y == other.y
            and self.player == other.player
        )

    def __hash__(self):
        return hash((self.x, self.y, self.player))
