from QNetwork import QNetwork
import torch
import torch.nn as nn
import torch.nn.functional as func


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


def main():
    state_dim = 2
    nb_actions = 3
    ep_size = 8
    net = Net1(state_dim, nb_actions)
    agent = QNetwork(net, state_dim, ep_size)

    # Memorize fool data
    agent.memorize(torch.rand(ep_size, state_dim), torch.randint(0, nb_actions, (ep_size, 1)),
                   torch.rand(ep_size, state_dim), torch.rand(ep_size, 1))
    agent.update()

    return 0


if __name__ == "__main__":
    main()