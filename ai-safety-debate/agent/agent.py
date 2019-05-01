from .mcts import mcts


class Agent:
    def __init__(self, precommit_label=None, agentStrength=1000):
        self.precommit_label = precommit_label
        self.mcts = mcts(iterationLimit=agentStrength)

    def get_initial_statement(self):
        if self.precommit_label is not None:
            return self.precommit_label
        else:
            raise NotImplementedError()

    def select_move(self, debate):
        """
        Returns next move the agent wants to make
        """
        currentState = debate.currentState
        assert not currentState.isTerminal()
        action = self.mcts.search(initialState=currentState)
        return action
