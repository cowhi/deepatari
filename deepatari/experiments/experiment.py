import logging
_logger = logging.getLogger(__name__)

import sys
import os
import numpy as np
import time
import importlib

# TODO: If we don't have the next line it will screw up logging (probably because of 'import gym' in AtariEnv)
from deepatari.envs import AtariEnv
from deepatari.tools import Statistics

class Experiment(object):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    Attributes:
        name (str): The name of the network object.
        args (argparse.Namespace): All settings either default or set via command line arguments.
        target_dir (str): Directory where all logs are stored.
        rng (mtrand.RandomState): Initialized Mersenne Twister pseudo-random number generator.
        env (Environment): The envirnoment in which the agent actuates.
        mem (Memory): The replay memory to save the experiences.
        net (Learner): Object of one of the Learner modules.
        agent (Agent): The agent that performes the learning.
        stats (Statistics): The stats module that attaches itself to the experiment objects.

    Note:
        All subclasses must inherite this class --> 'class NewClass(Experiment)'

    """

    def __str__(self):
        """ Overwrites the object.__str__ method.

        Returns:
            string (str): Important parameters of the object.
        """
        return "'name':" + str(self.name) + ", " + \
               "'target_dir':" + str(self.target_dir) + ", " + \
               "'env':" + str(self.env.name) + ", " + \
               "'agent':" + str(self.agent.name) + ", " + \
               "'mem':" + str(self.mem.name) + ", " + \
               "'net':" + str(self.net.name)

    def __init__(self, args, name, target_dir):
        """ Initializes an experiment.

        Args:
            args (argparse.Namespace): All settings either default or set via command line arguments.
            name (str): The name of the network object.
            target_dir (str): Directory where all logs are stored.

        Note:
            This class should never be initialized directly. Please use
            'super(NewClass, self).__init__(args, name, target_dir)' as the first
            line in 'NewClass.__init__'.
        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        self.name = name
        self.args = args
        self.target_dir = target_dir
        # Mersenne Twister pseudo-random number generator
        self.rng = np.random.RandomState(self.args.random_seed)

        try:
            EnvironmentClass = getattr(
                    __import__('deepatari.envs.' + self.args.env_type.lower(),
                            fromlist=[self.args.env_type]),
                    self.args.env_type)
        except ImportError as exc:
            sys.stderr.write("ERROR: " + self.args.env_type + "\n")
            sys.stderr.write("ERROR: " + str(exc) + "\n")
            sys.exit(1)
        self.env = EnvironmentClass(self.args, self.rng)

        try:
            MemoryClass = getattr(
                    __import__('deepatari.memory.' + self.args.memory_type.lower(),
                            fromlist=[self.args.memory_type]),
                    self.args.memory_type)
        except ImportError as exc:
            sys.stderr.write("ERROR: " + self.args.memory_type + "\n")
            sys.stderr.write("ERROR: " + str(exc) + "\n")
            sys.exit(1)
        self.mem = MemoryClass(self.args, self.env.avail_actions, self.rng)

        try:
            LearnerClass = getattr(
                    __import__('deepatari.learner.' + self.args.learner_type.lower(),
                            fromlist=[self.args.learner_type]),
                    self.args.learner_type)
        except ImportError as exc:
            sys.stderr.write("ERROR: " + self.args.learner_type + "\n")
            sys.stderr.write("ERROR: " + str(exc) + "\n")
            sys.exit(1)
        self.net = LearnerClass(self.env, self.args, self.rng)

        try:
            AgentClass = getattr(
                    __import__('deepatari.agents.' + self.args.agent_type.lower(),
                            fromlist=[self.args.agent_type]),
                    self.args.agent_type)
        except ImportError as exc:
            sys.stderr.write("ERROR: " + self.args.agent_type + "\n")
            sys.stderr.write("ERROR: " + str(exc) + "\n")
            sys.exit(1)
        self.agent = AgentClass(self.env, self.mem, self.net, self.args, self.rng)

        self.stats = Statistics(self.agent, self.net, self.mem, self.env, self.args, self.target_dir)

        _logger.info("%s" % str(self))
