import logging
_logger = logging.getLogger(__name__)

import os
import lasagne
from lasagne.layers import dnn
import numpy as np
import theano
import theano.tensor as T

from .learner import Learner

class DQNLasagne(Learner):
    """ This class is an implementation of the DQN network based on the Lasagne Theano framework.

    The modules that interact with the agent, the replay memory and the
    statistic calls are implemented here, taking the individual requirements
    of the Lasagne framework into account. The code is adapted from:
    https://github.com/spragunr/deep_q_rl

    Attributes:
        input_shape (tuple[int]): Dimension of the network input.
        dummy_batch (numpy.ndarray): Dummy batche used to calculate Q-values for single states.
        network (lasagne.layers.dense.DenseLayer): Structure of the neural network.
        target_network (lasagne.layers.dense.DenseLayer): Structure of the target neural network.
        states_shared (theano.sandbox.cuda.var.CudaNdarraySharedVariable): Interface to CUDA allocated array for states.
        followup_states_shared (theano.sandbox.cuda.var.CudaNdarraySharedVariable): Interface to CUDA allocated array for followup-states.
        rewards_shared (theano.sandbox.cuda.var.CudaNdarraySharedVariable): Interface to CUDA allocated array for rewards.
        actions_shared (theano.sandbox.cuda.var.CudaNdarraySharedVariable): Interface to CUDA allocated array for actions.
        terminals_shared (theano.sandbox.cuda.var.CudaNdarraySharedVariable): Interface to CUDA allocated array for terminal state indicators.
        params (list): Network parameter.
        observations (dict): Dictionary for all shared variables for the theano train function.
        _theano_train (theano.compile.function_module.Function): Theano function that implements a full training step for one mini-batch and returns the Q-values and the costs.
        _theano_get_Q (theano.compile.function_module.Function): Theano function that calculates one forward pass through the network for a mini-batch and returns the Q-value.
        callback (Statistics): Hook for the statistics object to pass train and test information.

    Note:
        More attributes of this class are defined in the base class Learner.
    """

    def __init__(self, env, args, rng, name = "DQNLasagne"):
        """ Initializes a network based on the Lasagne Theano framework.

        Args:
            env (AtariEnv): The envirnoment in which the agent actuates.
            args (argparse.Namespace): All settings either with a default value or set via command line arguments.
            rng (mtrand.RandomState): Initialized Mersenne Twister pseudo-random number generator.
            name (str): The name of the network object.

        Note:
            This function should always call the base class first to initialize
            the common values for the networks.
        """
        _logger.info("Initialize object of type " + str(type(self).__name__))
        super(DQNLasagne, self).__init__(env, args, rng, name)
        self.input_shape = (self.batch_size, self.sequence_length, args.frame_width, args.frame_height)
        self.dummy_batch = np.zeros(self.input_shape, dtype=np.uint8)
        lasagne.random.set_rng(self.rng)

        self.network = self._create_layer()

        # TODO: Load weights from pretrained network?!
        if not self.args.load_weights == None:
            self.load_weights(self.args.load_weights)

        if self.target_update_frequency > 0:
            self.target_network = self._create_layer()
            self._copy_theta()

        states = T.tensor4('states')
        followup_states = T.tensor4('followup_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        self.states_shared = theano.shared(
                np.zeros(self.input_shape, dtype=theano.config.floatX)
        )
        self.followup_states_shared = theano.shared(
                np.zeros(self.input_shape, dtype=theano.config.floatX)
        )
        self.rewards_shared = theano.shared(
                np.zeros((self.batch_size, 1), dtype=theano.config.floatX),
                broadcastable=(False, True)
        )
        self.actions_shared = theano.shared(
                np.zeros((self.batch_size, 1), dtype='int32'),
                broadcastable=(False, True)
        )
        self.terminals_shared = theano.shared(
                np.zeros((self.batch_size, 1), dtype='int32'),
                broadcastable=(False, True)
        )

        qvalues = lasagne.layers.get_output(
                self.network,
                self._prepare_network_input(states)
        )

        if self.target_update_frequency > 0:
            qvalues_followup_states = lasagne.layers.get_output(
                    self.target_network,
                    self._prepare_network_input(followup_states)
            )
        else:
            qvalues_followup_states = lasagne.layers.get_output(
                    self.network,
                    self._prepare_network_input(followup_states)
            )
            qvalues_followup_states = theano.gradient.disconnected_grad(qvalues_followup_states)

        targets = (rewards +
                (T.ones_like(terminals) - terminals) *
                self.discount_rate *
                T.max(qvalues_followup_states, axis=1, keepdims=True)
        )
        errors = targets - qvalues[
                T.arange(self.batch_size),
                actions.reshape((-1,))].reshape((-1, 1))

        if self.clip_error > 0:
            quadratic_part = T.minimum(abs(errors), self.clip_error)
            linear_part = abs(errors) - quadratic_part
            cost_function = T.sum(0.5 * quadratic_part ** 2 + self.clip_error * linear_part)
        else:
            cost_function = T.sum(0.5 * errors ** 2)

        self.params = lasagne.layers.helper.get_all_params(self.network)
        self.observations = {
            states: self.states_shared,
            followup_states: self.followup_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }

        self._set_optimizer(cost_function)

        if self.momentum > 0:
            self.optimizer = lasagne.updates.apply_momentum(
                    self.optimizer,
                    None,
                    self.momentum
            )
        _logger.debug("Compiling _theano_train")
        self._theano_train = theano.function(
                [],
                [cost_function, qvalues],
                updates=self.optimizer,
                givens=self.observations)
        _logger.debug("Compiling _theano_get_Q")
        self._theano_get_Q = theano.function(
                [],
                qvalues,
                givens={states: self.states_shared})

        self.callback = None
        _logger.debug("%s" % self)

    def _create_layer(self):
        """ Build a network consistent with the DeepMind Nature paper. """
        _logger.debug("Output shape = %d" % self.output_shape)
        l_in = lasagne.layers.InputLayer(
                shape=self.input_shape
        )
        l_conv1 = dnn.Conv2DDNNLayer(
                l_in,
                name='conv_layer_1',
                num_filters=32,
                filter_size=(8, 8),
                stride=(4, 4),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.HeUniform(),
                b=lasagne.init.Constant(.1)
        )
        l_conv2 = dnn.Conv2DDNNLayer(
                l_conv1,
                num_filters=64,
                filter_size=(4, 4),
                stride=(2, 2),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.HeUniform(),
                b=lasagne.init.Constant(.1)
        )
        l_conv3 = dnn.Conv2DDNNLayer(
                l_conv2,
                num_filters=64,
                filter_size=(3, 3),
                stride=(1, 1),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.HeUniform(),
                b=lasagne.init.Constant(.1)
        )
        l_hidden1 = lasagne.layers.DenseLayer(
                l_conv3,
                num_units=512,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.HeUniform(),
                b=lasagne.init.Constant(.1)
        )
        l_out = lasagne.layers.DenseLayer(
                l_hidden1,
                num_units=self.output_shape,
                nonlinearity=None,
                W=lasagne.init.HeUniform(),
                b=lasagne.init.Constant(.1)
        )
        return l_out

    def _set_optimizer(self, cost_function):
        """ Initializes the selected optimization algorithm.

        Args:
            cost_function (theano.tensor.var.TensorVariable): The cost function for the optimizer.
        """
        _logger.debug("Optimizer = %s" % str(self.args.optimizer))
        if self.args.optimizer == 'rmsprop':
            self.optimizer = lasagne.updates.rmsprop(
                    cost_function,
                    self.params,
                    self.learning_rate,
                    self.decay_rate,
                    self.rms_epsilon
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = lasagne.updates.sgd(
                    cost_function,
                    self.params,
                    self.learning_rate
            )
        else:
            raise ValueError("Unrecognized update for this kind of network: {}".format(self.args.optimizer))

    def _prepare_network_input(self, states):
        """ Normalizes the states from one minibatch.

        Args:
            states (numpy.ndarray): Mini-batch of states, shape=(batch_size,sequence_length,frame_width,frame_height)

        Returns:
            normalized_states (numpy.ndarray): State values divided by the maximim state value, shape=(batch_size,sequence_length,frame_width,frame_height)
        """
        _logger.debug("Normalizing input")
        return np.true_divide(states, self.grayscales)

    def _copy_theta(self):
        """ Copies the weights of the current network to the target network. """
        _logger.debug("Copying weights")
        all_params = lasagne.layers.helper.get_all_param_values(self.network)
        lasagne.layers.helper.set_all_param_values(self.target_network, all_params)

    def _share_state(self, state):
        """ Copies a state into an dummy batch and then the dummy batch to the shared states.

        Args:
            state(numpy.ndarray): Single state, shape=(sequence_length,frame_width,frame_height).
        """
        _logger.debug("Copy to GPU")
        self.dummy_batch[0, ...] = state
        self.states_shared.set_value(self.dummy_batch)

    def train(self, minibatch, epoch):
        """ Prepare, perform and document a complete train step for one mini-batch.

        Args:
            minibatch (numpy.ndarray): Mini-batch of states, shape=(batch_size,sequence_length,frame_width,frame_height).
            epoch (int): Current train epoch.
        """
        _logger.debug("Complete trainig step for one minibatch")
        states, actions, rewards, followup_states, terminals = minibatch
        self.states_shared.set_value(states)
        self.followup_states_shared.set_value(followup_states)
        self.actions_shared.set_value(actions.reshape((self.batch_size, 1)))
        self.rewards_shared.set_value(rewards.reshape((self.batch_size, 1)))
        self.terminals_shared.set_value(terminals.reshape((self.batch_size, 1)))
        cost, qvalues = self._theano_train()
        # get only max qvalue per state
        # make mean value of all max qvaluens
        qvalues_max_avg = np.mean(np.amax(qvalues, axis=0))
        self.update_iterations += 1
        if (self.target_update_frequency > 0 and
                self.update_iterations % self.target_update_frequency == 0):
            self._copy_theta()
            _logger.debug("Network update #%d: Cost = %s , Average Q-value: %s" % (self.update_iterations, str(np.sqrt(cost)), str(qvalues_max_avg)))
        # update statistics
        if self.callback:
            self.callback.from_learner(np.sqrt(cost), qvalues_max_avg)

    def get_Q(self, state):
        """ Calculates the Q-values for one mini-batch.

        Args:
            state(numpy.ndarray): Single state, shape=(sequence_length,frame_width,frame_height).

        Returns:
            q_values (numpy.ndarray): Results for first element of mini-batch from one forward pass through the network, shape=(self.output_shape,)
        """
        self._share_state(state)
        return self._theano_get_Q()[0]

    def save_weights(self, target_dir, epoch):
        """ Saves the current network parameters to disk.

        Args:
            target_dir (str): Directory where the network parameters are stored for each episode.
            epoch (int): Current epoch.
        """
        filename = "%s_%s_%s_%d.npz" % (str(self.args.game.lower()), str(self.args.net_type.lower()), str(self.args.optimizer.lower()), (epoch + 1))
        np.savez(os.path.join(target_dir, filename), *lasagne.layers.get_all_param_values(self.network))

    def load_weights(self, source_file):
        """ Loads the network parameters from a given file.

        Args:
            source_file (str): Complete path to a file with network parameters.
        """
        loaded_weights = np.load(str(source_file))
        lasagne.layers.set_all_param_values(self.network, loaded_weights)
