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
        state = debate.state
        action = self.mcts.search(initialState=state)
        return action

        agent_playing = debate.agent_playing
        possible_actions = debate.getPossibleActions(state)
        action = 0
        next_state, agent_playing, done = debate.get_next_state(
            state, action, agent_playing
        )
        if done:
            winner = debate.judge.evaluate_debate(next_state, debate.initial_statements)
        return possible_actions[0]
