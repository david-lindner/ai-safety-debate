import numpy as np
from debate_state import DebateState
from visualization import plot_image_mask


class Debate:
    def __init__(self, agents, judge, N_moves, sample):
        assert len(agents) == 2
        self.agents = agents
        self.judge = judge
        self.N_moves = N_moves
        self.sample = sample
        self.initial_statements = [agent.get_initial_statement() for agent in agents]
        # TODO
        self.currentState = DebateState(
            self.sample, self.initial_statements, self.judge, self.N_moves, 0
        )

    def resetToStartPosition(self):
        self.currentState = DebateState(
            self.sample, self.initial_statements, self.judge, self.N_moves, 0
        )

    def play(self):
        while not self.currentState.isTerminal():
            action = self.agents[self.currentState.currentPlayer].select_move(self)
            self.currentState = self.currentState.takeAction(action)
            # David: put plotting in here for debugging, want to remove this eventually
            plot_image_mask(self.currentState)  # 28 is the MNIST image size
        winner = self.judge.evaluate_debate(
            np.stack([self.currentState.mask.sum(axis=0), self.sample]),
            self.initial_statements,
        )
        return winner
