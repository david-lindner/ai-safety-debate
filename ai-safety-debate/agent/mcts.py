# code obtained from https://pypi.org/project/mcts/ (and modified slightly)
import time
import math
import random


def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode:
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.maximizerNode = state.maximizerNode()


class mcts:
    def __init__(
        self,
        timeLimit=None,
        iterationLimit=None,
        explorationConstant=1,  # 1 / math.sqrt(2),
        rolloutPolicy=randomPolicy,
    ):
        if timeLimit:
            if iterationLimit:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = "time"
        else:
            if iterationLimit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = "iterations"
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == "time":
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.select(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def select(self, node):
        """
        Iteratively finds the best child, then the best child of that child, and so on, until encountering a terminal
        node or a node that hasn't yet been fully expanded.
        :param node: a node in the MCTS tree, from which we start exploring
        :return: a tree node
        """
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        """
        Takes a node where not all actions have been explored yet.
        Takes one of the unexplored actions, and returns the resulting node.
        """
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        """
        For a terminal node "node" which produced reward "reward", updates all of the nodes above (increasing visit
        count and updating the cumulative reward.
        :param node: terminal node
        :param reward: reward obtained at this node
        :return: doesn't return anything, of course! :-)
        """
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        """
        Runs the Upper Confidence Bounds algorithm (selection algorithm for stochastic multi-armed bandit)
        :param node: where to select the action
        :param explorationValue: parameter controlling the exploration-exploitation trade-off
        :return: the action with the highest value + "potential"
        """
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = (
                node.maximizerNode * child.totalReward / child.numVisits
                + explorationValue
                * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            )
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        """
        Actions and nodes are of a different type.
        This takes a node and returns the action that leads to that node -- i.e. it serves as a kind of translation.
        :param root: parent node
        :param bestChild: children node
        :return: action that, when taken at the parent, leads to the child
        """
        for action, node in root.children.items():
            if node is bestChild:
                return action
