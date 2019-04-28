import numpy as np
from debate_state import DebateState


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
            sample, self.initial_statements, self.judge, self.N_moves, 0
        )

    def resetToStartPosition(self):
        self.currentState = DebateState(
            sample, self.initial_statements, self.judge, self.N_moves, 0
        )

    def play(self):
        while not self.currentState.isTerminal():
            action = self.agents[self.currentState.currentPlayer].select_move(self)
            self.currentState = self.currentState.takeAction(action)
        winner = self.judge.evaluate_debate(self.currentState, self.initial_statements)
        return winner
