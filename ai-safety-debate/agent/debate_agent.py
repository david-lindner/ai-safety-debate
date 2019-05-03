from .mcts import mcts


class DebateAgent:
    def __init__(self, precommit_label=None, agentStrength=1000):
        self.precommit_label = precommit_label
        self.mcts = mcts(iterationLimit=agentStrength)

    def get_initial_statement(self):
        return self.precommit_label

    def select_move(self, debate):
        """
        Returns next move the agent wants to make
        """
        current_state = debate.currentState
        assert not current_state.isTerminal()
        action = self.mcts.search(initialState=current_state)
        return action


class DebatePlayers:
    def __init__(self, our_agent, opp_agent, index_of_our_agent=1, our_name="truth", opp_name="liar"):
        assert index_of_our_agent in [0, 1]
        if index_of_our_agent == 0:
            self.agents = (our_agent, opp_agent)
            self.names = (our_name, opp_name)
        else:
            self.agents = (opp_agent, our_agent)
            self.names = (opp_name, our_name)
        self.index_of_our_agent = index_of_our_agent

    def increment_our_wincount(self, utility):
        """
        Returns 1 when truth wins or in case of a draw.
        Returns 0 if truth lost.
        """
        assert self.index_of_our_agent in [0,1]
        if self.index_of_our_agent == 0:
            if utility >=0:
                return 1
            else:
                return 0
        else:
            if utility <= 0:
                return 1
            else:
                return 0

    def print_debate_result(self, utility, true_label):
        """Prints out the result of the debate (for debugging/presentation purposes)."""
        if utility > 0:
            winner = self.names[0]
        elif utility < 0:
            winner = self.names[1]
        else:
            winner = "draw"

        if self.index_of_our_agent:
            if self.index_of_our_agent == 0:
                truth_utility = utility
            else:
                truth_utility = utility * (-1)
            print("True answer: {}, P1 answer: {}, P2 answer {}, Winner: {}, \"Our\" utility: {}".format(
                true_label, self.agents[0].precommit_label, self.agents[1].precommit_label, winner, truth_utility)
            )
        else:
            print("True label: {}, P1 answer: {}, P2 answer {}, Winner: {}, P1 utility: {}".format(
                true_label, self.agents[0].precommit_label, self.agents[1].precommit_label, winner, utility)
            )
