import logging
_logger = logging.getLogger(__name__)

import sys
import numpy as np

from .memory import Memory

class ReplayMemory(Memory):
    """ This class is an implementation of a simple replay memory.

    This memory is a simple stack and mini-batches get selected randomly from
    the whole memory of size memory_size. There are no preferences for
    obersavtions with more reward or higher probability.

    Note:
        More attributes of this class are defined in the base class Memory.
    """

    def __init__(self, args, avail_actions, rng, name = "ReplayMemory"):
        """ Initializes a simple replay memory.

        Args:
            args (argparse.Namespace): All settings either with a default value or set via command line arguments.
            avail_actions (int): Number of possible actions the agent can use.
            rng (mtrand.RandomState): initialized Mersenne Twister pseudo-random number generator.
            name (str): The name of the network object.

        Note:
            This function should always call the base class first to initialize
            the common attribute values of the replay memory.
        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        super(ReplayMemory, self).__init__(args, avail_actions, rng, name)
        _logger.debug("%s" % self)


    def add(self, action, reward, frame, terminal):
        """ Adds a full observation to the simple replay memory.

        Args:
            action (int): The action that was chosen.
            reward (int): The reward received after taking the action.
            frame (numpy.ndarray): The new frame received after taking the action, shape=(frame_height, frame_width).
            terminal (bool): The new terminal state indicator after taking the action.
        """
        _logger.debug("Observation at %d of %d" % (self.current, self.count))
        assert frame.shape == self.frame_dims
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.frames[self.current, ...] = frame
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def get_minibatch(self):
        """ Selects indices from the memory and gets the full observation for each of those indices.

        Returns:
            prestates (numpy.ndarray): Collected prestates of shape=(batch_size, sequence_length, frame_height, frame_width).
            actions (numpy.ndarray): Collected actions of shape=(batch_size,).
            rewards (numpy.ndarray): Collected rewards of shape=(batch_size,).
            poststates (numpy.ndarray): Collected poststates of shape=(batch_size, sequence_length, frame_height, frame_width).
            terminals (numpy.ndarray): Collected terminal state indicators of shape=(batch_size,).
        """
        _logger.debug("Size = %d" % self.batch_size)
        assert self.count > self.sequence_length
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                # sample one index (ignore states wraping over
                index = self.rng.randint(self.sequence_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.sequence_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last observation) can be terminal state!
                if self.terminals[(index - self.sequence_length):index].any():
                    continue
                # otherwise use this index
                break
            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self._get_frame_sequence(index - 1)
            self.poststates[len(indexes), ...] = self._get_frame_sequence(index)
            indexes.append(index)
        # copy actions, rewards and terminals with direct slicing
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]  # TODO: not indexes + 1 ??
        terminals = self.terminals[indexes]
        return self.prestates, actions, rewards, self.poststates, terminals
