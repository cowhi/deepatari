import logging
_logger = logging.getLogger(__name__)

import sys
import numpy as np

from .experiment import Experiment

class AtariExp(Experiment):
    """ This class is an implementation of an Atari experiment.

    The experiment organizes all objects and directs the training in an Atari
    Game playing scenario.

    Note:
        More attributes of this class are defined in the base class Experiment.
    """

    def __init__(self, args, name = "AtariExperiment", target_dir = None):
        """ Initializes an experiment in Atari game playing.

        Args:
            args (argparse.Namespace): All settings either with a default value or set via command line arguments.
            name (str): The name of the experiment object.
            target_dir (str): Directory where the network parameters are stored for each episode.

        Note:
            This function should always call the base class first to initialize
            the common values for the experiments.
        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        super(AtariExp, self).__init__(args, name, target_dir)


    def run(self):
        """ Run a complete experiment.

        Returns:
            success (bool): After a successfull run returns with True.
        """
        if self.args.fill_mem_size > 0:
            _logger.info("########## INITIALIZING ##########")
            self._reset_exp()
        # loop over epochs
        for epoch in xrange(self.args.epochs):
            _logger.info("########## EPOCH %d ##########" % (epoch+1))
            if self.args.train_steps:
                self.train(epoch)
            if self.args.test_steps:
                self.test(epoch)
        self.stats.close()
        return True

    def _reset_exp(self):
        """ Reset the stats and fill replay memory. """
        _logger.info("Set random moves in ReplayMemory: %d" % self.args.fill_mem_size)
        self.agent.phase = "init"
        self.stats.reset_epoch_stats()
        self.agent.populate_mem(self.args.fill_mem_size)
        self.stats.write_epoch_stats(0)

    def train(self, epoch):
        """ Reset the stats and call the agent train function.

        Args:
            epoch (int): Number of current epoch.
        """
        _logger.info("Training in epoch %d for %d steps" % (epoch+1, self.args.train_steps))
        self.agent.phase = "train"
        self.stats.reset_epoch_stats()
        self.agent.train(self.args.train_steps, epoch)
        self.stats.write_epoch_stats((epoch + 1))
        # TODO: add option if keep all networks or only best
        if not self.target_dir == None:
            self.net.save_weights(self.target_dir, epoch)

    def test(self, epoch):
        """ Reset the stats and call the agent test function.

        Args:
            epoch (int): Number of current epoch.
        """
        _logger.info("Testing in epoch %d for %d steps" % (epoch+1, self.args.test_steps))
        self.agent.phase = "test"
        self.stats.reset_epoch_stats()
        self.agent.test(self.args.test_steps, epoch)
        self.stats.write_epoch_stats((epoch + 1))
