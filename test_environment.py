"""
Tests the Environment class
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
from environment import Environment


def next_states_func(states, actions, time, device):
    step = 0.01
    angles = 2 * np.pi * actions.view(-1, 1) / nb_actions
    added = torch.cat((torch.cos(angles), torch.sin(angles)),
                        dim=1) * step
    return states + added


def euclidean_dist_squared(tensor1, tensor2):
    return torch.sum((tensor1 - tensor2) ** 2, 1)


def rewards_func(states, actions, time, device):
    objective = torch.full(states.size(), 0.5, device=device)
    next_states = next_states_func(states, actions, time, device)
    d0 = euclidean_dist_squared(states, objective)
    d1 = euclidean_dist_squared(next_states, objective)
    return d0 - d1


def final_states_func(states, time, device):
    if time > 70:
        return torch.ones(states.size()[0], device=device)
    else:
        return torch.zeros(states.size()[0], device=device)



class Net(nn.Module):
    def __init__(self, state_dim, nb_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, nb_actions)

    def forward(self, batch):
        batch = func.relu(self.fc1(batch))
        batch = func.relu(self.fc2(batch))
        batch = self.fc4(batch)
        return batch


if __name__ == "__main__":
    state_dim = 2
    nb_actions = 4 # To begin with

    net = Net(state_dim, nb_actions)
    env = Environment(state_dim, nb_actions, net,
                      next_states_func,
                      final_states_func,
                      rewards_func,
                      device=torch.device("cuda"))

    # Create some random initial states
    initial_states = torch.rand(3, state_dim)
    env.explore(initial_states)

    for _ in range(5):
        initial_states = torch.rand(3000, state_dim)
        env.clear_agent_memory()
        env.explore(initial_states, True)
        env.train_agent(epochs=1)

    plt.subplot(211)
    env.agent.show_training()
    plt.subplot(212)
    env.plot_trajectories(torch.rand(32, state_dim))
    print(env.agent.device)
