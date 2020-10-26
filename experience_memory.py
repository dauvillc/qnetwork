"""
@author Clément Dauvilliers 26-10-2020

Defines the Experience Memory data structure used by the QAgent object.
"""

import torch
import random


def mem_tensor_append(mem: torch.tensor, new_tensor: torch.tensor) -> torch.tensor:
    """
    Adds a tensor as a new row to a memory tensor
    :param mem: The 2D matrix to which the new tensor should be added
    :param new_tensor: The 2D tensor to add to the memory.
                       Needs to have its number of columns equal to the memory's
                       columns.
    :return: a reference to mem
    """
    return torch.cat((mem, new_tensor.view(1, -1)), dim=0)


def get_random_rows(matrix: torch.tensor, nb_rows: int) -> torch.tensor:
    """
    Returns a concatenation of nb_rows rows taken randomly from the given matrix,
    without replacement.
    :param matrix: a torch.tensor from which the rows should be taken from.
                   Needs to have matrix.size()[0] >= nb_rows.
    :param nb_rows       The number of rows to take from matrix and concatenate
    :return: A concatenation of nb_rows rows from matrix.
    """
    first_dim_size = matrix.size()[0];
    if first_dim_size < nb_rows:
        raise IndexError("Not enough rows in matrix of shape " + matrix.size()
                         + " to extract " + nb_rows + " rows.")
        return None
    return matrix[random.sample(range(matrix.size), nb_rows)]


class ExperienceMemory:
    """
    An experience memory is a structures used to memorize the exploration
    that a QAgent has been trough.
    A single experience is a (s, a, s', r) tuple where:
    - s is the starting state
    - a is the action taken by the agent
    - s' is the resulting state from taking action a in the state s
    - r is the reward the agent received from the environment for choosing action
        a in the state s.
    """

    def __init__(self):
        # need init is True until a first experience is added using memorize()
        self.need_init = True
        self.reward_mem = None
        self.state_mem = None
        self.action_mem = None
        self.nstate_mem = None

    def memorize(self, state: torch.tensor,
                 action: torch.tensor,
                 next_state: torch.tensor,
                 reward: torch.float32):
        """
        Memorizes a single experience.
        :param state:   Starting state
        :param action:  Decided action
        :param next_state    Resulting state
        :param reward   Real value, reward received from the environment
        """
        # If nothing had been stored previously, the memory is initialized
        # as containing only the values from the first experience.
        if self.need_init:
            self.state_mem = state.view(1, -1)
            self.action_mem = action.view(1, -1)
            self.nstate_mem = next_state.view(1, -1)
            self.reward_mem = torch.tensor([[reward]])
            self.need_init = False
        else:
            self.state_mem = mem_tensor_append(self.state_mem, state)
            self.action_mem = mem_tensor_append(self.action_mem, action)
            self.nstate_mem = mem_tensor_append(self.nstate_mem, next_state)
            self.reward_mem = torch.cat((self.reward_mem,
                                         torch.tensor(reward).view(1, -1)),
                                        dim=0)

    def random_batch(self, batch_size: int):
        """
        Returns a random batch of experiences stored in the memory without deleting
        them.
        :param batch_size Number of experiences in the returned batch.
        :return: four torch tensors of size :
                 - states batch: (batch_size, state_dim)
                 - actions batch: (batch_size, action_dim)
                 - next states batch: (batch_size, state_dim)
                 - rewards batch (batch_size, 1)
        """
        # Randomly select single experiences through their indexes in the memory.
        indxs = random.sample(range(self.state_mem.size()[0]), batch_size)
        return (self.state_mem[indxs],
                self.action_mem[indxs],
                self.nstate_mem[indxs],
                self.reward_mem[indxs])

    def clear(self):
        """
        Clears the memory.
        """
        self.__init__()

    def __str__(self):
        return "ExperienceMemory object:\n" \
               + "State memory:\n" + str(self.state_mem) \
               + "\nAction memory:\n" + str(self.action_mem) \
               + "\nNext state memory:\n" + str(self.nstate_mem) \
               + "\nReward memory:\n" + str(self.reward_mem) \
               + "\n"
