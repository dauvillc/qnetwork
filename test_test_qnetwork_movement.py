"""
Tests the test module test_qnetwork_movement.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from test_qnetwork_movement import test, next_state
from QNetwork import QNetwork


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


state_dim, nb_actions = 2, 5
movements = 10
step = 0.01

net = Net(state_dim, nb_actions)

agent = QNetwork(net, state_dim, movements, lr=0.1)

test(agent, 100, 2)
agent.train_on_memory(32, 5)