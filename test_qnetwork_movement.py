"""
The objective here is to teach the agent to reach a certain position on a grid.
"""


from QNetwork import QNetwork
import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np


class Net1(nn.Module):
    """
    Net used for the agent.
    """
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


def get_rewards(states: torch.tensor, actions: torch.tensor):
    """
    Reward function for specific (state, action) couples
    """
    rewards = torch.empty(actions.size())
    for i in range(states.size()[0]):
        nstate = next_state(states[i], actions[i])
        new_dist = distance_square([1, 1], nstate)
        dist = distance_square([1, 1], states[i])
        rewards[i] = dist - new_dist
    return rewards / np.sqrt(2)


def distance_square(case1, case2) -> float:
    """
    :param case1: A 2 items iterable corresponding to a position
    :param case2: Another position
    :return: The square of euclidian distance between the two positions
    """
    return (case1[0] - case2[0]) ** 2 + (case1[1] - case2[1]) ** 2


def final_reward(final_state: torch.tensor) -> float:
    """
    Defines the final reward from an episode.
    :param final_state: State in which the agent was at the end of the episode
    :return: A scalar corresponding to the final reward.
    """
    return - distance_square(final_state, [1, 1]) / 2


def next_state(state: torch.tensor, action: int):
    """
    :param state: 1D torch tensor corresponding to a game state
    :param action: index of the action decided by the agent
    :return: The next state as a 1D torch tensor
    """
    step = 0.005
    nstate = torch.tensor((state[0], state[1]))
    if action == 0:
        nstate[1] -= step
    elif action == 1:
        nstate[0] += step
    elif action == 2:
        nstate[1] += step
    elif action == 3:
        nstate[0] -= step
    return nstate


def main():
    # A state is defined as its x and y coordinates
    state_dim = 2

    # There a 5 possible actions: North, west, south, east, stay
    nb_actions = 5

    # Number of episodes
    nb_episodes = 5000

    # Number of movements allowed in a single episode
    movements = 100

    net = Net1(state_dim, nb_actions)
    agent = QNetwork(net, state_dim, movements, lr=0.1)

    for ep in range(nb_episodes):
        # Play a single episode

        # Create arrays to store the successive states and taken actions
        states = torch.empty((movements + 1, state_dim))  # + 1 to make space for the last state
        actions = torch.empty(movements, dtype=torch.int32)

        # Start with a random position
        states[0] = torch.rand(2)

        for step in range(movements):
            # Take action
            actions[step] = agent.decide(states[step].view(1, -1)).item()

            # Get next state
            states[step + 1] = next_state(states[step], actions[step])

        # Get rewards
        rewards = get_rewards(states[:-1], actions)

        # Memorize the episode
        agent.memorize_exploration(states, actions, rewards)

        # Train after the episode
        agent.update()

        # Clear the memory every 500 episodes
        if ep % 500 == 0:
            agent.clear_memory()

        print("Episodes completed: ", ep + 1, " / ", nb_episodes, "(",
              (ep + 1) * 100 / nb_episodes, "%)", end="\r")
        # print("Final position: ", states[-1], " | Initial: ", states[0])

    agent.plot_trajectory(torch.rand((10, 2)), next_state)
    agent.show_training()

    return 0


if __name__ == "__main__":
    main()