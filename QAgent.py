"""
@author Clément Dauvilliers 26-10-2020
Defines the QAgent object.
"""

from random import random, randint
import torch
from experience_memory import ExperienceMemory


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

    def __init__(self, nb_states: int, nb_actions: int,
                 epsilon_prob: float = 0.05, gamma=0.9,
                 lr=0.1,
                 batch_replay_size=1024):
        """
        :param nb_states:   Number of states reachable in the environment.
        :param nb_actions:  Number of possible actions. If the number of actions
                            differs depending on the state, should be the maximum
                            amount of actions.
        :param epsilon_prob: Epsilon probability. Defaults to 5%.
        :param gamma Discount factor.
        :param lr Learning rate
        :param batch_replay_size Size of batches to train on during updates.
        """
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Matrix containing Qvalues for every (s, a) couple
        self.Q = torch.zeros([nb_states, nb_actions], dtype=torch.float32)
        self.epsilon_prob = epsilon_prob

        # Discount Factor
        self.gamma = gamma

        # Learning rate
        self.lr = lr

        # Experience memory
        self.mem = ExperienceMemory()
        self.batch_replay_size = batch_replay_size

    def decide(self, state: int):
        """
        :param state: State index
        :return: The action a that is best according to the agent
                (The one that has the best QValue), or a random action
                with probability epsilon.
        """
        if random() < self.epsilon_prob:
            return randint(0, self.nb_actions - 1)
        return torch.argmax(self.Q[state])

    def memorize(self, state: int, action: int, next_state: int,
                 reward: torch.float32):
        """
        Stores an experience into the experience memory.
        :param state:
        :param action:
        :param next_state:
        :param reward:
        """
        self.mem.memorize(torch.tensor([[state]]),
                          torch.tensor([[action]]),
                          torch.tensor([[next_state]]),
                          reward)

    def update(self):
        """
        Updates the agent's Q values using experience replay.
        """
        states, actions, nstates, rewards = self.mem.random_batch(self.batch_replay_size)
        for s, a, ns, r in zip(states, actions, nstates, rewards):
            self.Q[s, a] = (1 - self.lr) * self.Q[s, a] \
                + self.lr * (r + self.gamma * torch.max(self.Q[ns]))