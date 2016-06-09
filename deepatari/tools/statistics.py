import logging
_logger = logging.getLogger(__name__)

import sys
import os
import csv
import time
import numpy as np

class Statistics(object):
    """ This class handles all statistics of an experiment.

    The class keeps the statistics running and saves certain paramaters at the
    end of an epoch to files. It's also responsible for generating nice graphs
    to evaluate the training progress.

    Attributes:
        STATS_AGENT_TRAIN (tuple): Defines a tuple of all relevant agent training parameters for evaluation.
        STATS_AGENT_TEST (tuple): Defines a tuple of all relevant agent testing parameters for evaluation.
        STATS_AGENT_NET (tuple): Defines a tuple of all relevant network training parameters for evaluation.
        name (str): The name of the statistic object.
        agent (Agent): The agent that performes the learning.
        net (Learner): Object of one of the Learner modules.
        mem (Memory): The replay memory to save the experiences.
        env (Environment): The envirnoment in which the agent actuates.
        target_dir (str): Location to save the stats.
        csv_file_train (file): The file were the training parameters are stored.
        csv_writer_train (writer): Converts the data into a delimited string to save in the file.
        csv_file_test (file): The file were the testing parameters are stored.
        csv_writer_test (writer): Converts the data into a delimited string to save in the file.
        time_start (time): Keeps track of the experiment start time.
        phase (str): Indicates the current phase of the experiment.
    """


    STATS_AGENT_TRAIN = (
            ("epoch","Epoch","#","int"),
            ("phase","Phase","",np.object_),
            ("n_steps_epoch","Steps per Epoch","#","int"),
            ("n_games","Games per Epoch","#","int"),
            ("n_steps_games_avg","Steps per Game (avg)","#",),
            ("n_steps_games_min","Steps per Game (min)","#","int"),
            ("n_steps_games_max","Steps per Game (max)","#","int"),
            ("reward_epoch","Reward per Epoch","","float"),
            ("reward_game_avg","Reward per Game (avg)","","float"),
            ("reward_game_min","Reward per Game (min)","","float"),
            ("reward_game_max","Reward per Game (max)","","float"),
            ("epsilon","Exploration Rate","","float"),
            ("n_steps_total","Steps Total","#","int"),
            ("replay_memory_size","Replay Memory Size","#","int"),
            ("q_avg_epoch","Q-Value per Epoch (avg)","","float"),
            ("cost_avg_epoch","Cost per Epoch (avg)","","float"),
            ("weight_updates","Network Weight Updates","#","int"),
            ("time_total","Time Total","s","float"),
            ("time_epoch","Time Epoch","s","float"),
            ("steps_per_second","Steps per Second","#","int")
    )

    STATS_AGENT_TEST = (
            ("epoch","Epoch","#","int"),
            ("phase","Phase","",np.object_),
            ("n_steps_epoch","Steps per Epoch","#","int"),
            ("n_games","Games per Epoch","#","int"),
            ("n_steps_games_avg","Steps per Game (avg)","#",),
            ("n_steps_games_min","Steps per Game (min)","#","int"),
            ("n_steps_games_max","Steps per Game (max)","#","int"),
            ("reward_epoch","Reward per Epoch","","float"),
            ("reward_game_avg","Reward per Game (avg)","","float"),
            ("reward_game_min","Reward per Game (min)","","float"),
            ("reward_game_max","Reward per Game (max)","","float"),
            ("epsilon","Exploration Rate","","float"),
            ("n_steps_total","Steps Total","#","int"),
            ("replay_memory_size","Replay Memory Size","#","int"),
            ("q_avg_epoch","Q-Value per Epoch (avg)","","float"),
            ("cost_avg_epoch","Cost per Epoch (avg)","","float"),
            ("weight_updates","Network Weight Updates","#","int"),
            ("time_total","Time Total","s","float"),
            ("time_epoch","Time Epoch","s","float"),
            ("steps_per_second","Steps per Second","#","int")
    )

    STATS_NET = (
            ("epoch","Epoch","#","int"),
            ("n_batch_update","Batch Update","#","int"),
            ("cost_current","Cost per Batch Update","","float"),
            ("cost_average","Cost Average","","float"),
            ("qvalue_average","Q-Value per Batch Update","","float"),
            ("epsilon","Exploration Rate","","float")
    )

    # TODO: adapt stats for training and testing
    # TODO: separate stats for network

    def __str__(self):
        """ Overwrites the object.__str__ method.

        Returns:
            string (str): Important parameters of the object.
        """
        return "'name':" + str(self.name) + ", " + \
               "'time_start':" + str(self.time_start)

    def __init__(self, agent, net, mem, env, args, target_dir):
        """ Initialize an statistics object.

        Args:
            agent (Agent): The agent that performes the learning.
            net (Learner): Object of one of the Learner modules.
            mem (Memory): The replay memory to save the experiences.
            env (Environment): Current environment, which provides information for the learner.
            args (argparse.Namespace): All settings either default or set via command line arguments.
            target_dir (str): Location to save the stats.

        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        self.name = "Observer"
        # attach statistics to agent
        self.agent = agent
        self.agent.callback = self
        # attach statistics to net
        self.net = net
        self.net.callback = self
        # make replay memory and environment available
        self.mem = mem
        self.env = env
        # make target dir available
        self.target_dir = target_dir
        # check directory for savin stats
        #if not os.path.isdir(target_dir):
        #    os.makedirs(target_dir)
        if not self.target_dir == None:
            # setup file for train stats
            self.csv_file_train = open(os.path.join(target_dir, "stats_agent_train.csv"), "wb")
            self.csv_writer_train = csv.writer(self.csv_file_train)
            self.csv_writer_train.writerow([stat[0] for stat in self.STATS_AGENT_TRAIN])
            self.csv_file_train.flush()
            # setup file for test stats
            self.csv_file_test = open(os.path.join(target_dir, "stats_agent_test.csv"), "wb")
            self.csv_writer_test = csv.writer(self.csv_file_test)
            self.csv_writer_test.writerow([stat[0] for stat in self.STATS_AGENT_TEST])
            self.csv_file_test.flush()
        # initialize timer
        self.time_start = time.clock()

        _logger.debug("%s" % str(self))

    def close(self):
        """ Closes the logfiles after the experiment. """
        _logger.debug("Closing logfiles")
        if not self.target_dir == None:
            #if self.agent.phase in ("train","random"):
            self.csv_file_train.close()
            #elif self.agent.phase == "test":
            self.csv_file_test.close()

    def reset_epoch_stats(self):
        """ Resets the parameters to initial values for each epoch. """
        _logger.debug("Resetting stats")
        self.time_epoch_start = time.clock()
        self.n_steps_epoch = 0
        self.n_games = 0
        self.n_steps_game = 0
        self.n_steps_games_avg = 0
        self.n_steps_games_min = sys.maxint
        self.n_steps_games_max = -sys.maxint - 1
        self.reward_epoch = 0
        self.reward_game = 0
        self.reward_game_avg = 0
        self.reward_game_min = sys.maxint
        self.reward_game_max = -sys.maxint - 1
        self.epsilon = 1
        self.cost_avg_epoch = 0
        self.q_avg_epoch = 0

    def from_agent(self, reward, terminal, epsilon):
        """ Handles the callbacks from the agent.

        Args:
            reward (int): The reward received after taking the action.
            terminal (bool): The new terminal state indicator after taking the action.
            epsilon (float): The current epsilon value.

        """
        _logger.debug("Callback from agent")
        self.reward_epoch += reward
        self.reward_game += reward
        self.n_steps_epoch += 1
        self.n_steps_game += 1
        self.epsilon = epsilon
        if terminal:
            self.n_games += 1
            self.reward_game_avg += float(self.reward_game - self.reward_game_avg) / self.n_games
            self.reward_game_min = min(self.reward_game_min, self.reward_game)
            self.reward_game_max = max(self.reward_game_max, self.reward_game)
            self.reward_game = 0
            self.n_steps_games_avg += float(self.n_steps_game - self.n_steps_games_avg) / self.n_games
            self.n_steps_games_min = min(self.n_steps_games_min, self.n_steps_game)
            self.n_steps_games_max = max(self.n_steps_games_max, self.n_steps_game)
            self.n_steps_game = 0


    def from_learner(self, cost_batch, q_avg_batch):
        """ Handles the callbacks from the learner.

        Args:
            cost_batch (float): Cost per batch.
            q_avg_batch (float): Average max Q-value per batch.
        """
        _logger.debug("Callback from net")
        self.cost_avg_epoch += (cost_batch - self.cost_avg_epoch) / self.net.update_iterations
        self.q_avg_epoch += (q_avg_batch - self.q_avg_epoch) / self.net.update_iterations


    def write_epoch_stats(self, epoch):
        """ Writes the stats for the current epoch to disk.

        Args:
            epoch (int): Current epoch.

        """
        _logger.debug("Epoch = %d" % epoch)
        time_current = time.clock()
        time_total = time_current - self.time_start
        time_epoch = time_current - self.time_epoch_start
        if time_epoch != 0:
            steps_per_second = int(self.n_steps_epoch / time_epoch)
        else:
            steps_per_second = 1
        if self.n_games == 0:
            self.n_games = 1
            self.reward_game_avg = self.reward_game
        '''
        # getting qvalue dynamics ??
        if self.validation_states is None and self.mem.count > self.mem.batch_size:
            # sample states for measuring Q-value dynamics
            prestates, actions, rewards, poststates, terminals = self.mem.getMinibatch()
            self.validation_states = prestates
        if self.validation_states is not None:
            qvalues = np.empty((self.net.output_shape, self.net.batch_size))
            for i, state in enumerate(self.validation_states):
                qvalues[:,i] = self.net.predict(state)
            maxqs = np.max(qvalues, axis=1)
            assert maxqs.shape[0] == qvalues.shape[0]
            meanq = np.mean(maxqs)
        else:
            meanq = 0
        '''
        if not self.target_dir == None:
            if self.agent.phase in ("train","random"):
                content = (
                        epoch,
                        self.agent.phase,
                        self.n_steps_epoch,
                        self.n_games,
                        self.n_steps_games_avg,
                        self.n_steps_games_min,
                        self.n_steps_games_max,
                        self.reward_epoch,
                        self.reward_game_avg,
                        self.reward_game_min,
                        self.reward_game_max,
                        self.epsilon,
                        self.agent.n_steps_total,
                        self.mem.count,
                        self.q_avg_epoch,
                        self.cost_avg_epoch,
                        self.net.update_iterations,
                        "{:.2f}".format(time_total),
                        "{:.2f}".format(time_epoch),
                        steps_per_second
                )
                self.csv_writer_train.writerow(content)
                self.csv_file_train.flush()
            elif self.agent.phase == "test":
                content = (
                        epoch,
                        self.agent.phase,
                        self.n_steps_epoch,
                        self.n_games,
                        self.n_steps_games_avg,
                        self.n_steps_games_min,
                        self.n_steps_games_max,
                        self.reward_epoch,
                        self.reward_game_avg,
                        self.reward_game_min,
                        self.reward_game_max,
                        self.epsilon,
                        self.agent.n_steps_total,
                        self.mem.count,
                        self.q_avg_epoch,     #was: meanq,
                        self.cost_avg_epoch,
                        self.net.update_iterations,
                        "{:.2f}".format(time_total),
                        "{:.2f}".format(time_epoch),
                        steps_per_second
                )
                self.csv_writer_test.writerow(content)
                self.csv_file_test.flush()

        _logger.info("n_games: %d, average_reward: %f, min_game_reward: %d, max_game_reward: %d, epsilon: %f, time_epoch: %ds, steps_per_second: %d" % (self.n_games, self.reward_game_avg, self.reward_game_min, self.reward_game_max, self.epsilon, time_epoch, steps_per_second))
