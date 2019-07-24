import numpy as np
import copy


class DebateState:
    """
    Keeps track of the "ground truth", the claims have made so far, the judge (for evaluation of the final result).
    """

    def __init__(
        self,
        sample,
        initial_statements,
        judge,
        player_order,  # a vector of player labels -- who acts in which turn
        moves_left,  # debate has to tell the state how many moves can we make
        allow_black_pixels,
    ):

        self.sample = sample
        self.mask = np.stack((np.zeros_like(sample), np.zeros_like(sample)))
        self.initial_statements = initial_statements
        self.judge = judge
        self.moves_left = moves_left
        self.allow_black_pixels=allow_black_pixels
        if player_order == None:
            raise Exception("Player order not specified")

        # player order is an array determinating which player should play each round
        # starting_player is not used anymore
        self.current_player = player_order[0]
        self.player_order = player_order

    def getPossibleActions(self):
        if self.allow_black_pixels:
            # Shows the not selected features.
            return np.where(self.mask.sum(axis=0) == 0)[0]
        else:
            # Shows the not selected and nonzero features.
            return np.where((self.mask.sum(axis=0) == 0) & (self.sample != 0))[0]

    def takeAction(self, action):
        """Adds the latest action to the mask and changes the player."""
        assert action in self.getPossibleActions()
        newState = copy.copy(self)
        newState.mask = np.copy(self.mask)
        newState.mask[newState.current_player, action] = 1
        newState.moves_left -= 1
        # to prevent potential unexpected behavior in terminal states, we explicitly set the player to None there
        if newState.moves_left == 0:
            newState.current_player = None
        else:
            moves_past = len(newState.player_order) - newState.moves_left
            if (
                newState.player_order[moves_past] != 0
                and newState.player_order[moves_past] != 1
            ):
                raise Exception("Player order elements can be only either 0 or 1 ")

            newState.current_player = newState.player_order[moves_past]
        return newState

    def maximizerNode(self):
        """This tells MCTS whether it should try to optimize from first players POV or from the second players POV."""
        # return +1 if the current player is a maximizer, -1 if they are a minimizer
        if self.current_player == 0:
            return 1
        elif self.current_player == 1:
            return -1
        else:
            return None

    def isTerminal(self):
        """This keeps track of whether the game has ended. Important for MCTS and for knowing when to get rewards."""
        assert self.moves_left >= 0
        return self.moves_left == 0

    def getReward(self):
        """Calls the judge (atm a neural network, but possibly more general thingy) to compute utility to pl. 1."""
        assert self.isTerminal()
        mask = self.mask.sum(axis=0)
        # the judge needs a "shape" of the mask (which pixels got revealed) and the values of those pixels.
        utility = self.judge.evaluate_debate(
            np.stack((mask, self.sample * mask), axis=1), self.initial_statements
        )
        assert -1 <= utility <= 1
        return utility
