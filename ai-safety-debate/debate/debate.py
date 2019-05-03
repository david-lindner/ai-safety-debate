import numpy as np
from .debate_state import DebateState

try:
    from util.visualization import plot_image_mask

    visualization_available = True
except ModuleNotFoundError:
    visualization_available = False


class Debate:
    """
    Keeps track of the current state of the debate, queries agents for actions, and calls a judge at the end.
    Debug parameter will print out the current state (and possibly do other magical stuff).
    Changing sides = False would make one agent make all 3 actions in a row, then the other. By default, they switch
    after each action.
    Specifying player order could even make this order custom.
    """
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

        # player order is an array determining which player should play each round
        if not player_order:
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
        self.current_state = DebateState(
            self.sample,
            self.initial_statements,
            self.judge,
            self.player_order,
            self.N_moves,
        )


    def play(self, full_report=False):
        """
        Runs the debate and returns its result.
        :param full_report: Turning this on makes the debate return a vector of probabilities (one for each label).
        :return: By default, this returns the utility of the first player in the debate, maximum is 1, minimum is -1.
        """
        while not self.current_state.isTerminal():
            action = self.agents[self.current_state.current_player].select_move(self)
            self.current_state = self.current_state.takeAction(action)
            if self.debug and visualization_available:
                plot_image_mask(self.current_state)
        mask = self.current_state.mask.sum(axis=0)
        if full_report:
            probabilities = self.judge.full_report(
                np.stack([mask, self.sample * mask], axis=1)
            )
            return probabilities
        else:
            utility = self.judge.evaluate_debate(
                np.stack([mask, self.sample * mask], axis=1),
                self.initial_statements,
            )
            return utility
