"""
@author Clément Dauvilliers - 15/12/2020
Defines the Environment class used to interact with a DQN agent.
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
from QNetwork import QNetwork


class Environment:
    """
    An Environment contains all information and functions needed to interact with
    a deep Q-Network. This includes explorations with decisions from the agent,
    computing the rewards and training the agent.
    """
    def __init__(self, state_dim: int, nb_actions: int,
                 net: torch.nn.Module,
                 next_state_func,
                 final_states_func,
                 rewards_func,
                 device=None):
        """
        :param state_dim: Dimensions needed to describe a state. For example a position
            on a plan will need state_dim = 2.
        :param nb_actions: Number of different possible actions at most.
        :param net: Torch neural network used by the agent. Should have
            its output dimension equal to nb_actions and its input dimension equal to state_dim.
        :param next_state_func: Function of signature (2D torch tensor, a: int, time: int, torch device)
                                --> 2D torch tensor
            which for a tensor S where S[i, :] is a state, returns a tensor NS where NS[i, :] is the state
            obtained by performing action a in state S[i, :]. Time indicates how many transitions have already
            taken place during the exploration.
        :param final_states_func: Function of signature (2D torch tensor, time: int, torch device)
                                --> 1D torch tensor
            which for a tensor S where S[i, :] is a state, returns a tensor F where F[i] == 1 iff S[i, :] is final
            and F[i] == 0 otherwise.
        :param rewards_func: Function of signature (2D torch tensor, a: int, time: int, torch device)
                                --> 1D torch tensor
            which for a tensor S where S[i, :] is a state, returns a tensor R where R[i] is the reward for taking
            action a in the state S[i, :]. Time indicates how many transitions have already taken place during
            the exploration.
        :param device: Torch device to be used for computations and training
        """
        self.state_dim = state_dim
        self.nb_actions = nb_actions

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.agent = QNetwork(net, state_dim, 32, device=self.device)
        self.next_state_func = next_state_func
        self.final_states_func = final_states_func
        self.rewards_func = rewards_func

    def explore(self, initial_states: torch.tensor, verbose=False):
        """
        Performs parallel explorations and lets the agent memorize them.
        :param initial_states: 2D Torch tensor of shape (number_of_explorations, state_dim) indicating
            initial states for every exploration.
        :param verbose: Indicates whether to print some insight about how the exploration's going.
        """
        states = initial_states

        # Make sure the tensors are on the right device
        states = states.to(self.agent.device)

        time = 0
        while states.size()[0] > 0:
            # Decides of an action for each exploration
            actions = self.agent.decide(states)

            # Computes the next states following the actions taken and the current time
            next_states = self.next_state_func(states, actions, time, self.device)

            # Compute which states are final and adds this indicator to the states tensor
            indicator = self.final_states_func(next_states, time, self.device)
            next_states_with_indic = torch.cat((next_states, indicator.view(-1, 1)), dim=1)

            # Computes the rewards
            rewards = self.rewards_func(states, actions, time, self.device)

            # Memorizes all those steps into the agent
            self.agent.memorize(states, actions, next_states_with_indic, rewards)

            # Recreates a tensor of states with the new states where they are not final
            states = next_states[indicator == 0]
            time += 1

            if verbose:
                print(str(time) + " episode(s) completed - Explorations still running: " + str(states.size()[0]),
                      end='\r')

    def train_agent(self, epochs=1, verbose=True):
        """
        Trains the agent over its memorized experiences.
        :param epochs: Number of times the agent trains over its whole memory.
        :param verbose: Whether to print some insight about the training.
        """
        self.agent.train_on_memory(self.agent.batch_size, epochs)

    def set_agent_parameters(self, batch_size=None, lr=None, epsilon=None, discount=None):
        """
        Sets parameters for the DQN agent. Any given unspecified parameter will remain
        unchanged.
        :param batch_size: size of batches used during the training sessions.
        :param lr: Learning rate.
        :param epsilon: Epsilon probability: probability of taking a random action rather than the
            most rewarding one.
        :param discount: Discount factor.
        """
        self.agent.batch_size = batch_size
        self.agent.set_learning_rate(lr)
        self.agent.discount = discount
        self.agent.epsilon = epsilon

    def set_agent_device(self, device: torch.device):
        """
        Sets a torch device to use for the agent computations (preferably the GPU)
        :param device: Torch device object
        """
        self.agent.set_device(device)

    def clear_agent_memory(self):
        """
        Clears the agent's exploration memory
        """
        self.agent.clear_memory()

    def plot_trajectories(self, initial_states):
        """
        ONLY AVAILABLE IF STATE DIM IS 1 OR 2.
        Plots the trajectory of the agent starting from the given initial states
        on a 2D (if self.state_dim == 1) or 3D (if self.state_dim == 2) graph.
        :param initial_states: (N, state_dim) torch tensor indicating the starting states
        """
        initial_states = initial_states.to(self.device)
        for state in initial_states:
            state = state.view(1, self.state_dim)
            time = 0
            indicator = 0
            states_mem = []
            while indicator == 0:

                # Decides of an action
                actions = self.agent.decide(state)

                # Computes the next state following the actions taken and the current time
                next_state = self.next_state_func(state, actions, time, self.device)

                indicator = self.final_states_func(next_state, time, self.device).item()

                states_mem.append(next_state.view(self.state_dim))
                state = next_state

                time += 1

            # Plotting
            if self.state_dim == 1:
                plt.plot(torch.arange(0, time), [s.item() for s in states_mem])
            elif self.state_dim == 2:
                plt.plot([s[0] for s in states_mem], [s[1] for s in states_mem], "-")
                plt.plot([states_mem[0][0]], [states_mem[0][1]], "go")
                plt.plot([states_mem[-1][0]], [states_mem[-1][1]], "ro")
