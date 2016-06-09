import logging
_logger = logging.getLogger(__name__)

import gym
import sys
import os
import numpy as np
import cv2

from .env import Environment

class AtariEnv(Environment):
    """ This class is an implementation of an Atari environment.

    We are using the OpenAI Gym version, but pimp it a little with direct access
    to some settings from the original ALE implementation.

    The environment communicates the possible actions to the agent and provides
    an interface to interact and observe changes in the state of the game.

    Attributes:
        ALL_ACTIONS (Dict): Dictionary that has the action ID as keys and the action name as values.
        game (str): Name of the game to load into the emulator.
        gym (str): OpenAI Gym environment with the selected game.
        train_all (bool): Indicates if the network uses all possible actions as output or only the available ones.
        avail_actions (tuple[int]): The IDs of the availabe actions.
        n_avail_actions (int): Number of available actions for the agent to select for a specific environment.
        avail_actions_indices (tuple): Array with size of n_avail_actions with numbers from n
        counts_lives (bool): Indicates if the game has more than one live until game over.
        just_lost_live (bool): Indicates if the agent has just lost a live.
        current_lives (int): Number of available lives until game over.
        current_frame (tuple[int]): The current frame as provided by the gym environment.
        terminal_state (bool): Indicates if game over or not.
        frame_dims (tuple[int]): Height and width of the current frame.

    Note:
        More attributes of this class are defined in the base class Environment.
    """

    ALL_ACTIONS = {
            0 : "NOOP",
            1 : "FIRE",
            2 : "UP",
            3 : "RIGHT",
            4 : "LEFT",
            5 : "DOWN",
            6 : "UPRIGHT",
            7 : "UPLEFT",
            8 : "DOWNRIGHT",
            9 : "DOWNLEFT",
            10 : "UPFIRE",
            11 : "RIGHTFIRE",
            12 : "LEFTFIRE",
            13 : "DOWNFIRE",
            14 : "UPRIGHTFIRE",
            15 : "UPLEFTFIRE",
            16 : "DOWNRIGHTFIRE",
            17 : "DOWNLEFTFIRE",
            }

    def __str__(self):
        """ Overwrites the object.__str__ method.

        Returns:
            string (str): Important parameters of the object.
        """
        return "'name':" + str(self.name) + ", " + \
               "'game':" + str(self.game) + ", " + \
               "'actions':" + str(self.avail_actions) + ", " + \
               "'actions':" + str((self.gym).get_action_meanings()) + ", " + \
               "'n_avail_actions':" + str(self.n_avail_actions)

    def __init__(self, args, rng, name = "OpenAIGym"):
        """ Initializes the Atari environment.

        Args:
            args (argparse.Namespace): All settings either with a default value or set via command line arguments.
            rng (mtrand.RandomState): Initialized Mersenne Twister pseudo-random number generator.
            name (str): The name of the environment object.

        Note:
            This function should always call the base class first to initialize
            the common values for all environments.
        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        super(AtariEnv, self).__init__(name)

        self.game = args.game
        self.gym = gym.make(self.game)
        (self.gym).ale.setInt('random_seed', rng.randint(666))
        (self.gym).ale.setInt('frame_skip', args.frame_skip)
        (self.gym).ale.setFloat('repeat_action_probability', args.repeat_action_probability)
        (self.gym).ale.setBool('color_averaging', args.color_averaging)
        self.train_all = args.train_all
        self.avail_actions = (self.gym)._action_set
        self.n_avail_actions = len(self.avail_actions)
        #self.avail_actions_indices = np.arange(len(self.avail_actions))
        if (self.gym).ale.lives() == 0:
            self.counts_lives = False
        else:
            self.counts_lives = True
            self.just_lost_live = False
            self.current_lives = (self.gym).ale.lives()
        self.current_frame = None
        self.terminal_state = None
        # OpenCV expects width as first and height as second s
        self.frame_dims = (args.frame_width, args.frame_height)
        _logger.debug("%s" % self)

    def reset_env(self):
        """ Resets the game parameters to start a new game. """
        _logger.debug("Resetting environment and setting terminal=False")
        self.current_frame = self.gym.reset()
        self.terminal_state = False
        if self.counts_lives:
            self.just_lost_live = False
            self.current_lives = (self.gym).ale.lives()

    def step(self, action):
        """ Perform an action and observe the resulting state.

        Args:
            action (int): Selected action ID to perform in envirnoment.

        Returns:
            reward (float): The change of score after performing the action.

        """
        _logger.debug("Getting index of action %s" % (str(action)))
        action = np.where(self.avail_actions == action)[0][0]
        self.current_frame, reward, self.terminal_state, info = self.gym.step(action)
        self.current_frame = cv2.resize(cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY), self.frame_dims)
        if self.counts_lives:
            self.just_lost_live = self.has_just_lost_live()
        return reward

    def get_current_frame(self):
        """ Check if there actually is current frame and if so return it.

        Returns:
            current_frame (tuple[int]): The current frame as provided by the gym environment.
        """
        _logger.debug("Dims = %s" % str(self.frame_dims))
        assert self.current_frame is not None
        return self.current_frame

    def is_state_terminal(self):
        """ Check if the terminal state indicator is set and if so return it.

        Returns:
            terminal_state (bool): Indicates if game over or not.
        """
        _logger.debug("terminal = %s" % str(self.terminal_state))
        assert self.terminal_state is not None
        return self.terminal_state

    def has_just_lost_live(self):
        """ Check if the agent has just lost a live.

        Returns:
            just_lost_live (bool): Indicates if the agent has just lost a live.
        """
        _logger.debug("%d > %d ?" % (self.current_lives, (self.gym).ale.lives()))
        if self.current_lives > (self.gym).ale.lives():
            self.current_lives = (self.gym).ale.lives()
            return True
        return False
