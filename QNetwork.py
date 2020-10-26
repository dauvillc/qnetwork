"""
@author Clément Dauvilliers 26-10-2020
Defines the QAgent object.
"""

import torch


class StateIndexOutOfRangeError(Exception):
    """
    Error raised when the user gives a state to the QAgent which is out of
    the (0, nb_states - 1) range.
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class QAgent:
    """
    A QAgent represents a QLearning Agent, which approximates
    the optimal QValue of every state, action couple (s, a).
    """

    def __init__(self, nb_states: int, nb_actions: int, epsilon_prob: float=0.05):
        """
        :param nb_states:   Number of states reachable in the environment.
        :param nb_actions:  Number of possible actions. If the number of actions
                            differs depending on the state, should be the maximum
                            amount of actions.
        :param epsilon_prob: Epsilon probability. Defaults to 5%.
        """
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Matrix containing Qvalues for every (s, a) couple
        self.Q = torch.zeros([nb_states, nb_actions], dtype=torch.float32)
        self.epsilon_prob = epsilon_prob

        # Experience memory
        self.

    def decide(self, state: int):
        """
        :param state: State index
        :return: The action a that is best according to the agent
                (The one that has the best QValue), or a random action
                with probability epsilon.
        """
