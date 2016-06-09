import logging
_logger = logging.getLogger(__name__)
import os
import numpy as np

from .learner import Learner

class DQNKeras(Learner):
    """ This class is an implementation of the DQN network based on Keras.
    The modules that interact with the agent, the replay memory and the
    statistic calls are implemented here, taking the individual requirements
    of the Keras framework into account.
    Attributes:
    Note:
        More attributes of this class are defined in the base class Learner.
    """

    def __init__(self, env, args, rng, name = "DQNKeras"):
        """ Initializes a network based on Keras.
        Args:
            env (AtariEnv): The envirnoment in which the agent actuates.
            args (argparse.Namespace): All settings either with a default value or set via command line arguments.
            rng (mtrand.RandomState): initialized Mersenne Twister pseudo-random number generator.
            name (str): The name of the network object.
        Note:
            This function should always call the base class first to initialize
            the common values for the networks.
        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        super(DQNKeras, self).__init__(env, args, rng, name)


    def _train(self, minibatch, epoch):
        """ Prepare, perform and document a complete train step for one mini-batch
        Args:
            minibatch (numpy.ndarray): Mini-batch of states, shape=(batch_size,sequence_length,frame_width,frame_height)
            epoch (int): Current train epoch
        """
        pass

    def _get_Q(self, state):
        """" Calculates the Q-values for one mini-batch.
        Args:
            state(numpy.ndarray): Single state, shape=(sequence_length,frame_width,frame_height).
        Returns:
            q_values (numpy.ndarray): Results for first element of mini-batch from one forward pass through the network, shape=(self.output_shape,)
        """
        pass

    def _save_weights(self, target_dir, epoch):
        """ Saves the current network parameters to disk.
        Args:
            target_dir (str): Directory where the network parameters are stored for each episode.
            epoch (int): Current epoch.
        """
        pass

    def _load_weights(self, source_file):
        """ Loads the network parameters from a given file.
        Args:
            source_file (str): Complete path to a file with network parameters.
        """
        pass
