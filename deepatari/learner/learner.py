import logging
_logger = logging.getLogger(__name__)
import os
import numpy as np
import abc

class Learner(object):
    """ This is the base class for all learning algorithms.

    The modules that interact with the agent, the replay memory and
    statistics are called here and forward to the respective internal
    modules in the subclasses.

    Attributes:
        grayscales (int): Maximum of color value per pixel for normalization.
        name (str): The name of the network object.
        frame_dims (tuple[int]): Dimension of the resized frames.
        output_shape (int): Dimension of the output layer.
        sequence_length (int): Determines how many frames form a state.
        batch_size (int): Size of the mini-batch for one learning step.
        discount_rate (float): Determines the discount of future rewards.
        learning_rate (float): Learning rate of the network.
        decay_rate (float): Decay rate for RMSProp and Adadelta algorithms.
        rms_epsilon (float): Epsilon for RMSProp.
        momentum (float): Momentum for optimizer.
        clip_error (float): Clip error term in update between this number and its negative to avoid gradient become zero.
        target_update_frequency (int): Copy weights of training network to target network after this many steps.
        update_iterations (int): Counter for target network updates.
        min_reward (float): Lower boundary for rewards.
        max_reward (float): Upper boundary for rewards.
        rng (mtrand.RandomState): Initialized Mersenne Twister pseudo-random number generator.

    Note:
        All subclasses must inherite this class --> 'class NewClass(Learner)'.

    """
    __metaclass__ = abc.ABCMeta

    grayscales = 255

    def __str__(self):
        """ Overwrites the object.__str__ method.

        Returns:
            string (str): Important parameters of the object.
        """
        return "'name':" + str(self.name) + ", " + \
               "'input_shape':" + str(self.input_shape) + ", " + \
               "'output_shape':" + str(self.output_shape) + ", " + \
               "'optimizer':" + str(self.args.optimizer)

    def __init__(self, env, args, rng, name):
        """ Initializes a learner object.

        Args:
            env (Environment): Current environment, which provides information for the learner.
            args (argparse.Namespace): All settings either default or set via command line arguments.
            rng (mtrand.RandomState): Initialized Mersenne Twister pseudo-random number generator.
            name (str): The name of the network object.

        Note:
            This class should never be initialized directly. Please use
            'super(NewClass, self).__init__(env, args, rng)' as the first
            line in 'NewClass.__init__'.
        """
        self.name = name
        self.args = args
        self.frame_dims = (self.args.frame_width, self.args.frame_height)
        if args.train_all:
            self.output_shape = len(env.ALL_ACTIONS)
        else:
            self.output_shape = env.n_avail_actions
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.learning_rate = args.learning_rate #epsilon
        self.decay_rate = args.decay_rate # rho
        self.rms_epsilon = args.rms_epsilon
        self.momentum = args.momentum
        self.clip_error = args.clip_error
        self.target_update_frequency = args.target_update_frequency
        self.update_iterations = 0
        self.min_reward = args.min_reward
        self.max_reward = args.max_reward
        self.rng = rng

    @abc.abstractmethod
    def train(self):
        """ Individual train function for each subclass. """
        pass

    @abc.abstractmethod
    def get_Q(self):
        """ Get results from one forward pass through the network. """
        pass

    @abc.abstractmethod
    def save_weights(self):
        """ Save the network weights acording to the used library. """
        pass

    @abc.abstractmethod
    def load_weights(self):
        """ Save the network weights acording to the used library. """
        pass
