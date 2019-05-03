import numpy as np
from .debate_state import DebateState

try:
    from util.visualization import plot_image_mask

    visualization_available = True
except ModuleNotFoundError:
    visualization_available = False


class Debate:
    def __init__(
        self,
        agents,
        judge,
        N_moves,
        sample,
        debug=False,
        changing_sides=True,
        player_order=None,
    ):
        assert len(agents) == 2
        self.agents = agents
        self.judge = judge
        self.N_moves = N_moves
        self.sample = sample
        self.initial_statements = [agent.get_initial_statement() for agent in agents]
        self.debug = debug

        # player order is an array determinating which player should play each round
        if player_order is None:
            if not changing_sides:
                if N_moves % 2 != 0:
                    raise Exception(
                        "The number of rounds in simultaneous debate should be even"
                    )
                player_order = [0 for i in range(N_moves // 2)] + [
                    1 for i in range(N_moves // 2)
                ]
            else:
                player_order = [i % 2 for i in range(N_moves)]

        elif len(player_order) != N_moves:
            raise Exception("player order should have the same length as debate")

        self.player_order = player_order
        self.currentState = DebateState(
            self.sample,
            self.initial_statements,
            self.judge,
            self.N_moves,
            0,
            self.player_order,
        )

    def resetToStartPosition(self):
        self.currentState = DebateState(
            self.sample, self.initial_statements, self.judge, self.N_moves, 0
        )

    def play(self, full_report=False):
        while not self.currentState.isTerminal():
            action = self.agents[self.currentState.currentPlayer].select_move(self)
            self.currentState = self.currentState.takeAction(action)
            if self.debug and visualization_available:
                plot_image_mask(self.currentState)
        if full_report:
            probabilities = self.judge.evaluate_debate(
                np.stack([self.currentState.mask.sum(axis=0), self.sample], axis=1)
            )
            return probabilities
        else:
            utility = self.judge.evaluate_debate(
                np.stack([self.currentState.mask.sum(axis=0), self.sample], axis=1),
                self.initial_statements,
            )
            return utility
