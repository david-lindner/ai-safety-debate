class Agent:
    def __init__(self, precommit_label=None):
        self.precommit_label = precommit_label

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
        agent_playing = debate.agent_playing
        possible_actions = debate.get_possible_actions(state)
        action = 0
        next_state, agent_playing, done = debate.get_next_state(
            state, action, agent_playing
        )
        if done:
            winner = debate.judge.evaluate_debate(next_state, debate.initial_statements)
        return possible_actions[0]
