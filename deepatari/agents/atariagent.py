import logging
_logger = logging.getLogger(__name__)

import sys
#import random
import numpy as np

from .agent import Agent
from .ataristatebuffer import AtariStateBuffer

class AtariAgent(Agent):
    """ This class is an implementation of an Atari agent.

    The agent interacts with the given environment, organizes the trainig of the
    network and sends information to the statistics.

    Attributes:
        buf (AtariStateBuffer): Simple buffer of sequence_length to concatenate frames to form the current state.
        n_avail_actions (int): Number of available actions for the agent to select for a specific environment.
        avail_actions (tuple[int]): The IDs of the availabe actions.
        train_all (bool): Indicates if the network uses all possible actions as output or only the available ones.
        random_starts (int): Perform max this number of dummy actions at beginning of an episode to produce more random game dynamics.
        sequence_length (int): Determines how many frames form a state.
        epsilon_start (float): Start value of the exploration rate (epsilon).
        epsilon_end (float): Final value of the exploration rate (epsilon).
        epsilon_decay_steps (int): Number of steps from epsilon_start to epsilon_end.
        epsilon_test (float): Exploration rate (epsilon) during the test phase.
        train_frequency (int): Perform training after this many game steps.
        train_repeat (int): Number of times to sample minibatch during training.

    Note:
        More attributes of this class are defined in the base class Agent.
    """

    def __init__(self, env, mem, net, args, rng, name = "AtariAgent"):
        """ Initializes an agent for the Atari environment.

        Args:
            env (AtariEnv): The envirnoment in which the agent actuates.
            mem (ReplayMemory): The replay memory to save the experiences.
            net (Learner): Object of one of the Learner modules.
            args (argparse.Namespace): All settings either with a default value or set via command line arguments.
            rng (mtrand.RandomState): initialized Mersenne Twister pseudo-random number generator.
            name (str): The name of the network object.

        Note:
            This function should always call the base class first to initialize
            the common values for the networks.
        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        super(AtariAgent, self).__init__(env, mem, net, args, rng, name)

        self.buf = AtariStateBuffer(args)
        self.n_avail_actions = self.env.n_avail_actions
        self.avail_actions = self.env.avail_actions
        self.train_all = args.train_all
        self.random_starts = args.random_starts
        self.sequence_length = args.sequence_length

        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_test = args.epsilon_test

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat

        _logger.debug("%s" % self)

    def _do_dummy_steps(self):
        """ Do some dummy steps at the beginning of each new episode for better randomization. """
        _logger.debug("Restarting environment with a number of dummy actions")
        self.env.reset_env()
        for i in xrange(self.rng.randint(self.sequence_length, self.random_starts) + 1):
            reward = self.env.step(0)
            frame = self.env.get_current_frame()
            terminal = self.env.is_state_terminal()
            assert not terminal, "terminal state occurred during random initialization"
            # add dummy states to buffer
            self.buf.add(frame)

    def _update_epsilon(self):
        """ Update the exploration rate (epsilon) with regard to the decay rate

        Returns:
            epsilon (float): Upated epsilon value.
        """
        _logger.debug("Updating exploration rate")
        if self.n_steps_total < self.epsilon_decay_steps:
            return self.epsilon_start - self.n_steps_total * (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        else:
            return self.epsilon_end

    def step(self, epsilon):
        """ Perform one step in the environment, send the results to the buffer and update the stats.

        Args:
            epsilon (float): The current epsilon value.
        """
        _logger.debug("Epsilon %f " % epsilon)
        if self.rng.random_sample() < epsilon:
            how = "random"
            action = self.rng.choice(self.avail_actions)
        else:
            # if not random choose action with highest Q-value
            how = "predicted"
            state = self.buf.get_current_state()
            qvalues = self.net.get_Q(state)
            _logger.debug("qvalues shape = %s, type = %s" % (str(qvalues.shape),str(type(qvalues))))
            assert len(qvalues.shape) == 1, "Qvalues not as expected -> " + qvalues.shape
            if self.train_all:
                qvalues = qvalues[np.array(self.avail_actions)]
            action = self.avail_actions[np.argmax(qvalues)]
            #_logger.debug("action %s <-- Qvalues: %s" % (str(action),str(qvalues)))
        # perform the action
        reward = self.env.step(action)
        frame = self.env.get_current_frame()
        self.buf.add(frame)
        terminal = self.env.is_state_terminal()
        _logger.debug("Observation: action=%s (%s), reward=%s, frame_dims=%s, just_lost_live=%s, terminal=%s" % (str(action), str(how), str(reward), str(frame.shape), str(self.env.just_lost_live), str(terminal) ))

        # TODO: check if lost live to end episode
        #if self.has_just_lost_live:

        # restart the game if over
        if terminal:
            #_logger.debug("GAME OVER: reached terminal state --> restarting")
            self._do_dummy_steps()

        # call callback to record statistics
        if self.callback:
            self.callback.from_agent(reward, terminal, epsilon)

        return action, reward, frame, terminal

    def populate_mem(self, size):
        """ Play a given number of steps to prefill the replay memory

        Args:
            size (int): The desired size of the memory initialization.
        """
        _logger.debug("Playing without exploitation for %d steps " % size)
        for i in xrange(size):
            action, reward, frame, terminal = self.step(1)
            self.mem.add(action, reward, frame, terminal)

    def train(self, steps, epoch):
        """ Performs a complete training epoch, filling the replay memory and calling the network train function.

        Args:
            steps (int): The number of steps.
            epoch (int): The current epoch.
        """
        _logger.debug("Training epoch %d for %d steps" % ((epoch + 1), steps))
        for i in xrange(steps):
            # perform game step
            action, reward, frame, terminal = self.step(self._update_epsilon())
            self.mem.add(action, reward, frame, terminal)
            # train after every train_frequency steps
            if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
                for j in xrange(self.train_repeat):
                    states, actions, rewards, followup_states, terminals = self.mem.get_minibatch()

                    if not self.train_all:
                        actions = np.asarray(
                                [np.where(self.avail_actions == action)[0][0] for action in actions],
                                dtype = np.uint8)
                    # train the network
                    minibatch = states, actions, rewards, followup_states, terminals
                    self.net.train(minibatch, epoch)
            # increase number of training steps for epsilon decay
            self.n_steps_total += 1

    def test(self, steps, epoch):
        """ Performs a complete testing epoch.

        Args:
            steps (int): The number of steps.
            epoch (int): The current epoch.
        """
        # just make sure there is sequence_length frames to form a state
        _logger.debug("Testing epoch %d for %d steps" % ((epoch + 1), steps))
        self._do_dummy_steps()
        for i in xrange(steps):
            self.step(self.epsilon_test)

    def play(self, num_games):
        """ Plays the game for a num_games times.

        Args:
            num_games (int): The number of games to play until stop.
        """
        _logger.debug("Playing without exploration for %d games " % num_games)
        self._do_dummy_steps()
        for i in xrange(num_games):
            terminal = False
            while not terminal:
                action, reward, frame, terminal = self.step(self.epsilon_test)
