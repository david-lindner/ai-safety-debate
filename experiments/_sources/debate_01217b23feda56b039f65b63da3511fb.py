import numpy as np
from visualization import plot_image_mask


class Debate:
    def __init__(self, agents, judge, N_moves, sample):
        assert len(agents) == 2
        self.agents = agents
        self.judge = judge
        self.N_moves = N_moves
        self.state = np.stack((np.zeros_like(sample), sample))
        self.agent_playing = 0
        self.initial_statements = [agent.get_initial_statement() for agent in agents]
        self.done = False

    def reset(self):
        self.state[0, :] = 0
        self.agent_playing = 0
        self.done = False

    def play(self):
        while not self.done:
            action = self.agents[self.agent_playing].select_move(self)
            self.state, self.agent_playing, self.done = self.get_next_state(
                self.state, action, self.agent_playing
            )
            #plot_image_mask(self.state, 28)
        winner = self.judge.evaluate_debate(self.state, self.initial_statements)
        return winner

    def get_possible_actions(self, state):
        # not selected and nonzero feature
        return np.where((state[0] == 0) & (state[1] != 0))[0]

    def get_next_state(self, state, action, agent_playing):
        state = np.copy(state)
        state[0, action] = 1
        new_agent_playing = (agent_playing + 1) % 2
        done = np.sum(state[0]) >= self.N_moves
        return state, new_agent_playing, done
