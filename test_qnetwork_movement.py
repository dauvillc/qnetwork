"""
Implements a simple test for the QNetwork class where the agent learns to reach a central
position, starting at a random initial position
"""


from QNetwork import QNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_rewards(states: torch.tensor, actions: torch.tensor, step: float, device=None):
    """
    Reward function for specific (state, action) couples
    """
    if device is None:
        device = torch.device("cpu")
    obj = torch.ones((states.size()[0], 2)) / 2
    nstates = torch.zeros(states.size(), device=device)
    for i in range(states.size()[0]):
        nstates[i] = next_state(states[i], actions[i], step, device)
    rewards = distance_square(states, obj) - distance_square(nstates, obj)
    return rewards / np.sqrt(2)


def distance_square(cases1, cases2) -> float:
    """
    :return: The square of euclidian distance between the two positions,
             line per line.
    """
    return (cases1[:, 0] - cases2[:, 0]) ** 2 + (cases1[:, 1] - cases2[:, 1]) ** 2


def final_reward(final_state: torch.tensor) -> float:
    """
    Defines the final reward from an episode.
    :param final_state: State in which the agent was at the end of the episode
    :return: A scalar corresponding to the final reward.
    """
    return - distance_square(final_state, [1, 1]) / 2


def next_state(state: torch.tensor, action: int, step: float, device=None):
    """
    :param state: 1D torch tensor corresponding to a game state
    :param action: index of the action decided by the agent
    :param step: Distance travelled at each move
    :param device: Device to use for computations, defaults to the CPU
    :return: The next state as a 1D torch tensor
    """
    if device is None:
        device = torch.device("cpu")
    nstate = torch.tensor((state[0], state[1]), device=device)
    if action == 0:
        nstate[1] -= step
    elif action == 1:
        nstate[0] += step
    elif action == 2:
        nstate[1] += step
    elif action == 3:
        nstate[0] -= step
    return nstate


def test(agent: QNetwork, movements=100, nb_episodes=1000, step=0.01, show_plots=True):
    """
    Tests the ability of the QNetwork to learn to reach the position (0.5, 0.5)
    while spawning at random coordinates in [0, 1]^2.
    :param agent: QNetwork to be tested. Needs to have state_dim == 2 and 5 possible actions.
    :param movements: Number of moves the agent is allowed to have
    :param step: Distance travelled at each move
    :param nb_episodes: Number of episodes on which the agent trains
    :param show_plots: if True, the agent will plot the results of the training
    :return: The agent's loss memory
    """
    # A state is defined as its x and y coordinates
    state_dim = 2

    # Calculation device
    device = torch.device("cpu")

    # net = Net1(state_dim, nb_actions)
    # QNetwork(net, state_dim, movements, lr=0.1, device=torch.device("cpu"))

    for ep in range(nb_episodes):
        # Play a single episode

        # Create arrays to store the successive states and taken actions
        states = torch.empty((movements + 1, state_dim), device=device)  # + 1 to make space for the last state
        actions = torch.empty(movements, dtype=torch.int32, device=device)

        # Start with a random position
        states[0] = torch.rand(2)

        for move in range(movements):
            # Take action
            actions[move] = agent.decide(states[move].view(1, -1)).item()

            # Get next state
            states[move + 1] = next_state(states[move], actions[move], step, device)

        # Get rewards
        rewards = get_rewards(states[:-1], actions, step, device)

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

    if show_plots:
        plt.figure("Training summary")
        plt.subplot(211)
        plt.title("Agent Trajectories")
        agent.plot_trajectory(torch.rand((50, 2)), lambda s, a: next_state(s, a, step, device))
        plt.subplot(212)
        plt.title("MSE Loss")
        agent.show_training()
        plt.show()
    return agent.loss_mem

    return 0
