"""
@author Clément Dauvilliers 29-10-2020
Defines the QNetwork object.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import matplotlib.pyplot as plt
from progress_bar import printProgressBar
from experience_memory import ExperienceMemory


class QNetwork:
    """
    A QNetwork is a neural network which uses a Q-Learning approach associated with
    Deep Learning to learn to approximate a reward function over a given state-action couple.
    A QNetwork object is not the neural network itself but rather a tool to make it work with
    Q Deep Learning.
    """

    def __init__(self, neural_net: nn.Module, state_dim: int,
                 batch_size: int, lr=0.01, epsilon_prob=0.05, discount=0.9,
                 device=None):
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
        :param device:   Device that will be used to compute the calculations. Defaults to the the
                         first gpu if possible, or the CPU otherwise.
        """
        self.net = neural_net
        self.net.zero_grad()
        self.state_dim = state_dim
        self.forward = self.net.forward
        self.batch_size = batch_size
        self.epsilon = epsilon_prob
        self.discount = discount

        # If the user did not specify a computation device, the cpu is used by default
        # This is because GPU isn't necessarily faster for explorations
        if device is None:
            dev_name = "cpu"
            device = torch.device(dev_name)

        # Set the neural network and memory to the device
        self.net.to(device)
        self.mem = ExperienceMemory(device)

        self.device = device

        # Training memory
        self.loss_mem = []

        # Update tools
        self.optimizer = optim.SGD(self.net.parameters(), lr)

    def memorize(self, states: torch.tensor, actions: torch.IntTensor, next_states: torch.tensor,
                 rewards: torch.tensor):
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

    def memorize_exploration(self, states: torch.tensor,
                             actions: torch.IntTensor,
                             rewards: torch.tensor):
        """
        Memorizes a whole exploration process with a final single reward.
        Should be used for processes for which the reward isn't specifically known for
        every state-action couple, but rather according to a final score.
        :param states: Successive states encountered. Should be a tensor of shape
                      (number_of_states, state_dim). This should include
        :param actions: Successive actions decided by the agent. Should be a tensor of shape
                       (number_of_states - 1, )
        :param next_states: For each state-action (s, a) encountered, state s' returned by the
                           environment. Same shape as :param state:.
        :param rewards (number_of_states - 1, )-sized 1D Tensor indicating the rewards for the episode
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = states[1:].clone().detach()
        self.mem.memorize(states[:-1], actions, next_states, rewards)

    def set_last_rewards(self, nb_experiences: int, reward: torch.double):
        """
        Sets the rewards for the last memorized experiences to a given value.
        This should be used for example when the reward is not known for every specific
        (state, action) couple, but can be deduced from the final state reached: Use this function
        to set the rewards for the episode to the final reward.
        :param nb_experiences: number of experiences whose rewards should be affected
        :param reward: scalar indicating to which value the last rewards should be set
        """
        self.mem.set_last_rewards(nb_experiences, reward)

    def decide(self, states: torch.tensor):
        """
        Decides which action is best for a given batch of states.
        :param states (Batch_size, state_dim) set of states.
        :return: A (Batch_size, 1) int tensor A where A[i, 0] is the index of the decided action.
        """
        # Make sure the states tensor runs on the right device
        states = states.to(self.device)

        dice = torch.rand(states.size()[0], device=self.device)
        output = self.forward(states)
        random_actions = torch.randint(0, output.size()[1], (states.size()[0],), device=self.device)
        actions = torch.argmax(output, dim=1).type(torch.int64)
        return actions * (dice >= self.epsilon) + random_actions * (dice < self.epsilon)

    def clear_memory(self):
        """
        Clears the agent's Experience Memory.
        """
        self.mem.clear()

    def train_on_batch(self, states, actions, next_states, rewards):
        """
        Trains the network on a batch of experiences
        :param states: (batch_size, state_dim) tensor indicating states
        :param actions: (batch_size, 1) int tensor indicating actions taken
        :param next_states: (batch_size, state_dim) tensor indicating next states
        :param rewards: (batch_size, 1) float tensor indicating
        """

        """
        The Target value to compute the loss is taken as
        y = reward + discount * max {Q[next_state, a'] for all action a'}
        Since we do not have that maximum value, we use the network's estimation.
        """

        output = self.forward(states).gather(1, actions.view(states.size()[0], 1)).view((-1,))
        max_next_qval = self.forward(next_states).max(1)[0]

        # Modify the target so that Y[k, a] = r  + gamma * max_net_val and Y[k, a'] is unchanged for a' != a
        target = rewards + self.discount * max_next_qval
        target = target.detach()

        # Compute the loss
        loss = func.mse_loss(output, target)
        self.loss_mem.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self):
        """
        Updates the QNetwork's parameters using its experience memory.
        """
        # Get a random batch from the experience memory
        states, actions, next_states, rewards = self.mem.random_batch(self.batch_size)
        self.train_on_batch(states, actions, next_states, rewards)

    def train_on_memory(self, batch_size, epochs):
        """
        Trains the agent on experiences from its experience replay memory.
        :param batch_size: Batch size for training
        :param epochs: Number of times the mem should be fully browsed
        """
        print("Training on ", epochs, " epochs from the replay memory..")

        # Get all data from the replay memory
        states, actions, next_states, rewards = self.mem.all()

        # Split them into batches
        states_batches = torch.split(states, batch_size)
        actions_batches = torch.split(actions, batch_size)
        next_states_batches = torch.split(next_states, batch_size)
        rewards_batches = torch.split(rewards, batch_size)

        # Number of batches
        nb_batches = len(states_batches)

        # Train
        for ep in range(epochs):
            batches_completed = 0
            for states, actions, next_states, rewards \
                    in zip(states_batches, actions_batches, next_states_batches, rewards_batches):
                self.train_on_batch(states, actions, next_states, rewards)
                batches_completed += 1
                printProgressBar(batches_completed, nb_batches,
                                 "Epoch " + str(ep + 1) + "/" + str(epochs), length=90)

    def show_training(self):
        """
        Plots the training metrics.
        """
        plt.plot([self.batch_size * (i + 1) for i in range(len(self.loss_mem))], self.loss_mem)
        plt.xlabel("Batches")
        plt.ylabel("MSE Loss")

    def plot_trajectory(self, initial_states: torch.tensor, next_state_function,
                        steps=100):
        """
        ONLY AVAILABLE IF STATE DIM IS 1 OR 2.
        Plots the trajectory of the agent starting from the given initial states
        on a 2D (if self.state_dim == 1) or 3D (if self.state_dim == 2) graph.
        :param initial_states: (N, state_dim) torch tensor indicating the starting states
        :param next_state_function: Function used to determine the next state.
            Should have signature (state: torch.tensor, action: int)
        :param steps: Number of successive states that should be plotted.
        """
        # Make sure the initial state runs on the right device
        initial_states = initial_states.to(self.device)

        if self.state_dim != 1 and self.state_dim != 2:
            raise ValueError("State dimension too large to plot agent trajectory.\n")
        for initial_state in initial_states:
            states = torch.empty((steps, self.state_dim))
            states[0] = initial_state

            # Exploration
            for step in range(steps - 1):
                action = self.decide(states[step].view(1, -1)).item()
                states[step + 1] = next_state_function(states[step], action)

            # Plotting
            if self.state_dim == 1:
                plt.plot(torch.arange(0, step), states)
                plt.plot([0], [initial_state[0]], "go")
                plt.plot([steps - 1], [states[-1].item()], "ro")
            elif self.state_dim == 2:
                plt.plot(states[:, 0], states[:, 1])
                plt.plot([initial_state[0]], [initial_state[1]], "go")
                plt.plot([states[-1, 0]], [states[-1, 1]], "ro")

    def set_learning_rate(self, new_lr: float):
        """
        Sets a value for the network's learning rate.
        :param new_lr: New value for the learning rate
        """
        self.optimizer.lr = new_lr
