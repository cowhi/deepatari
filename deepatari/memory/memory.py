import logging
_logger = logging.getLogger(__name__)
import os
import numpy as np
import abc

class Memory(object):
    """ This is the base class for all memory implementations.

    The agent stores its experiences in the memory and the learner uses it as a
    source for training examples.

    Attributes:
        name (str): The name of the network object.
        memory_size (int): Maximum elements in replay memory.
        actions (numpy.ndarray): Action memory allocation of shape=(memory_size,).
        rewards (numpy.ndarray): Rewards memory allocation of shape=(memory_size,).
        frames (numpy.ndarray): Frame memory allocation of shape=(memory_size, frame_height, frame_width).
        terminals (numpy.ndarray): Terminal state indicator memory allocation of shape=(memory_size,).
        sequence_length (int): Determines how many frames form a state.
        frame_dims (tuple[int]): Dimension of the resized frames.
        batch_size (int): Size of the mini-batch for one learning step.
        count (int): Current number of elements in the replay memory.
        current (int): Current index in the replay memory.
        train_all (bool): Use all possible actions or minimum set for training.
        avail_actions (int): Number of possible actions the agent can use.
        prestates (numpy.ndarray): Prestate memory allocation of shape=(batch_size, sequence_length, frame_height, frame_width).
        poststates (numpy.ndarray):Poststate memory allocation of shape=(batch_size, sequence_length, frame_height, frame_width).
        rng (mtrand.RandomState): Initialized Mersenne Twister pseudo-random number generator.

    Note:
        All subclasses must inherite this class --> 'class NewClass(Memory)'

    """
    __metaclass__ = abc.ABCMeta

    def __str__(self):
        """ Overwrites the object.__str__ method.

        Returns:
            string (str): Important parameters of the object.
        """
        return "'name':" + str(self.name) + ", " + \
               "'size':" + str(self.memory_size) + ", " + \
               "'sequence_length':" + str(self.sequence_length) + ", " + \
               "'frame_dims':" + str(self.frame_dims) + ", " + \
               "'batch_size':" + str(self.batch_size) + ", " + \
               "'count':" + str(self.count) + ", " + \
               "'current':" + str(self.current) + ", " + \
               "'prestates_dims':" + str(self.prestates.shape) + ", " + \
               "'poststates_dims':" + str(self.poststates.shape)


    def __init__(self, args, avail_actions, rng, name):
        """ Initialize the learner super class.

        Args:
            env (Environment): Current environment, which provides information for the learner.
            args (argparse.Namespace): All settings either default or set via command line arguments.
            rng (mtrand.RandomState): Initialized Mersenne Twister pseudo-random number generator.

        Note:
            This class should never be initialized directly. Please use
            'super(NewClass, self).__init__(args, avail_actions, rng, name)' as the first
            line in 'NewClass.__init__'

        """
        self.name = name
        self.memory_size = args.memory_size
        # preallocate memory
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.frames = np.empty((self.memory_size, args.frame_height, args.frame_width), dtype = np.uint8)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.sequence_length = args.sequence_length
        self.frame_dims = (args.frame_height, args.frame_width)
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0
        self.train_all = args.train_all
        self.avail_actions = avail_actions
        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.sequence_length) + self.frame_dims, dtype = np.uint8)
        self.poststates = np.empty((self.batch_size, self.sequence_length) + self.frame_dims, dtype = np.uint8)
        self.rng = rng

    @abc.abstractmethod
    def add(self):
        """ Individual add function for each subclass. """
        pass

    @abc.abstractmethod
    def get_minibatch(self):
        """ Individual get_minibatch function for each subclass. """
        pass

    def _get_frame_sequence(self, index):
        """ Selects the last frames of sequence_length ending with the element at index.

        Args:
            index (int): Index of selected state in replay memory.

        Returns:
            state (numpy.ndarray): Stacked frames from replay memory of shape=(sequence_length, frame_width, frame_height)
        """

        _logger.debug("Index = %d" % index)
        assert self.count > self.sequence_length, "replay memory has not enough frames use at least --random_steps 5 (= sequence_length + 1)"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        if index >= self.sequence_length - 1:
            return self.frames[(index - (self.sequence_length - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.sequence_length))]
            return self.frames[indexes, ...]
