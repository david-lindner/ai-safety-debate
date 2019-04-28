from mcts import mcts


class Agent:
    def __init__(self, precommit_label=None, agentStrength=1000):
        self.precommit_label = precommit_label
        self.mcts = mcts(iterationLimit=agentStrength)

    def get_initial_statement(self):
        if self.precommit_label:
            return self.precommit_label
        else:
            raise NotImplementedError()

    def select_move(self, debate):
        """
        Returns next move the agent wants to make
        """
        # TODO mcts magic here
        # state = debate.state
        # action = self.mcts.search(initialState=state)
        # return action

        possible_actions = debate.get_possible_actions(debate.state)
        action = possible_actions[0]
        print("action", action)
        return action
