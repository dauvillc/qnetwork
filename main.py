from QAgent import QAgent
from random import randint
import matplotlib.pyplot as plt
import torch


def get_next_state(s, a, grid_size):
    x, y = to_coords(s, grid_size)
    if a == 0:
        y = max(y - 1, 0)
    elif a == 1:
        x = min(x + 1, grid_size - 1)
    elif a == 2:
        y = min(y + 1, grid_size - 1)
    elif a == 3:
        x = max(x - 1, 0)
    return to_index((x, y), grid_size)


def dist(x, y):
    return (x - 10) ** 2 + (y - 10) ** 2


def get_reward(s, a, grid_size):
    x, y = to_coords(s, grid_size)
    x2, y2 = to_coords(get_next_state(s, a, grid_size), grid_size)
    return (dist(x, y) - dist(x2, y2)) / 10000


def to_coords(index, grid_size):
    return (int)(index / grid_size), index % grid_size


def to_index(coords, grid_size):
    return (int)(coords[0] * grid_size + coords[1])


if __name__ == '__main__':
    nb_states = 1600
    nb_actions = 5
    grid_size = 40

    batch_size = 32

    agent = QAgent(nb_states, nb_actions, batch_replay_size=batch_size, lr=0.5)

    episodes = 10000
    success_map = torch.zeros((grid_size, grid_size))
    for ep in range(episodes):
        # Random starting position
        start_x, start_y = randint(0, grid_size - 1), randint(0, grid_size - 1)
        curr_state = to_index((start_x, start_y), grid_size)

        # Give the agent 80 steps to reach (10, 10)
        for _ in range(batch_size):
            # Explore and memorize
            taken_act = agent.decide(curr_state)
            # reward = get_reward(curr_state, taken_act, grid_size)
            next_state = get_next_state(curr_state, taken_act,  grid_size)

            agent.memorize(curr_state, taken_act, next_state, 0)

            curr_state = next_state

        # Write the final distance in the success map
        final_dist = dist(*to_coords(curr_state, grid_size))
        success_map[start_x, start_y] = final_dist
        agent.mem.set_last_rewards(batch_size, -final_dist / 1000)

        agent.update()

        if ep % 1000 == 0:
            print("Completed ", ep, " episodes.\n")
            agent.mem.clear()

    fig = plt.figure()
    plt.imshow(success_map)
    plt.show()