from QNetwork import QNetwork
import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt


class Net1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, batch):
        batch = func.relu(self.fc1(batch))
        batch = func.relu(self.fc2(batch))
        batch = self.fc3(batch)
        return batch


def reward(states, actions):
    return (actions == 0) * 1 + (actions != 0) * (-1)


def main():
    state_dim = 2
    nb_actions = 3
    ep_size = 8
    net = Net1(state_dim, nb_actions)
    agent = QNetwork(net, state_dim, ep_size)

    for _ in range(100):
        # Memorize fool data
        states = torch.rand(ep_size, state_dim)
        actions = agent.decide(states)
        next_states = torch.rand(ep_size, state_dim)
        rewards = reward(states, actions)
        agent.memorize(states, actions,
                       next_states, reward(states, actions))
        agent.update()
        # agent.clear_memory()

    # agent.show_training()

    # Displaying the agent's decisions
    states_interv = torch.linspace(0, 1, 100)
    states_grid = torch.cartesian_prod(states_interv, states_interv)
    actions = agent.decide(states_grid)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(states_grid[:, 0], states_grid[:, 1], actions)
    plt.show()

    return 0


if __name__ == "__main__":
    main()