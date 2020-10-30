"""
@author Clément Dauvilliers 29-10-2020
Defines the QNetwork object.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from experience_memory import ExperienceMemory


class QNetwork:
    """
    A QNetwork is a neural network which uses a Q-Learning approach associated with
    Deep Learning to learn to approximate a reward function over a given state-action couple.
    A QNetwork object is not the neural network itself but rather a tool to make it work with
    Q Deep Learning.
    """
    def __init__(self, neural_net: nn.Module, state_dim: int,
                 batch_size: int, lr=0.01, epsilon_prob=0.05, discount=0.99):
        """
        :param neural_net: A Neural Network created with PyTorch. Needs to be a subclass of
                           torch.nn.Module and implement methodes __init__ and forward(self, batch).
        :param state_dim: Number of dimensions needed to define a state. Needs to equal the input dimension
                          of the given neural net.
        :param batch_size: Number of experiences on which the network trains during each update.
                           NOTE that the network has to explore at least batch_size experiences
                           before training a first time.
        :param lr:   Learning rate.
        :param epsilon_prob: Probability that the network chooses a random action rather than the
                             best one according to the QValues. Only relevant if decide() is used.
        :param discount: Discount factor (usually called gamma), representing the importance of early
                         decisions comparatively to later ones.
        """
        self.net = neural_net
        self.net.zero_grad()
        self.state_dim = state_dim
        self.mem = ExperienceMemory()
        self.forward = self.net.forward
        self.batch_size = batch_size
        self.epsilon = epsilon_prob
        self.discount = discount

        # Update tools
        self.optimizer = optim.SGD(self.net.parameters(), lr)
        self.loss = nn.MSELoss()

    def memorize(self, states: torch.tensor, actions: torch.IntTensor, next_states: torch.tensor, rewards: torch.tensor):
        """
        Memorizes a sequence of experiences which can be trained on later.
        An experience is a (s, a, ns, r) tuple where:
        -s is the starting state;
        -a is the decided action;
        -ns is the state resulting from taking action a in state s;
        -r is the reward received from the environment.
        :param states: A 2D (batch_size, state_dim) shaped tensor containing the experiences' states.
        :param actions: A 2D (batch_size, 1) integer tensor containing the experiences' decided actions.
        :param next_states: A 2D (batch_size, state_dim) tensor containing the experiences' next_states.
        :param rewards: A 2D (batch_size, 1) tensor containing the experiences' rewards.
        """
        for s, a, ns, r in zip(states, actions, next_states, rewards):
            self.mem.memorize(s, a.item(), ns, r.item())

    def clear_memory(self):
        """
        Clears the agent's Experience Memory.
        """
        self.mem.clear()

    def update(self):
        """
        Updates the QNetwork's parameters using its experience memory.
        """
        # Get a random batch from the experience memory
        states, actions, next_states, rewards = self.mem.random_batch(self.batch_size)

        """
        The Target value to compute the loss is taken as
        y = reward + discount * max {Q[next_state, a'] for all action a'}
        Since we do not have that maximum value, we use the network's estimation.
        """
        output = self.forward(states)
        max_next_qval = torch.max(self.forward(next_states), dim=1)
        target = output.clone().detach()

        # Modify the target so that Y[k, a] = r  + gamma * max_net_val and Y[k, a'] is unchanged for a' != a
        target[range(self.batch_size), actions] = rewards + self.discount * max_next_qval.values

        # Compute the loss
        loss = self.loss(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
