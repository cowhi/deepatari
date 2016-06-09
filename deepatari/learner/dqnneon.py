import logging
_logger = logging.getLogger(__name__)

from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Gaussian
from neon.optimizers import RMSProp, Adam, Adadelta
from neon.layers import Affine, Conv, GeneralizedCost
from neon.transforms import Rectlin
from neon.models import Model
from neon.transforms import SumSquared
from neon.util.persist import save_obj
import numpy as np
import os
import sys

from .learner import Learner


class DQNNeon(Learner):
    """ This class is an implementation of the DQN network based on Neon.

    The modules that interact with the agent, the replay memory and the
    statistic calls are implemented here, taking the individual requirements
    of the Lasagne framework into account. The code is adapted from:
    https://github.com/tambetm/simple_dqn

    Attributes:
        input_shape (tuple[int]): Dimension of the network input.
        dummy_batch (numpy.ndarray): Dummy batche used to calculate Q-values for single states.
        batch_norm (bool): Indicates if normalization is wanted for a certain layer (default=False).
        be (neon.backends.nervanagpu.NervanaGPU): Describes the backend for the Neon implementation.
        input (neon.backends.nervanagpu.GPUTensor): Definition of network input shape.
        targets(neon.backends.nervanagpu.GPUTensor): Definition of network output shape.
        model (neon.models.model.Model): Generated Neon model.
        target_model (neon.models.model.Model): Generated target Neon model.
        cost_func (neon.layers.layer.GeneralizedCost): Cost function for model training.
        callback (Statistics): Hook for the statistics object to pass train and test information.

    Note:
        More attributes of this class are defined in the base class Learner.
    """

    def __init__(self, env, args, rng, name = "DQNNeon"):
        """ Initializes a network based on the Neon framework.

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
        super(DQNNeon, self).__init__(env, args, rng, name)
        self.input_shape = (self.sequence_length,) + self.frame_dims + (self.batch_size,)
        self.dummy_batch = np.zeros((self.batch_size, self.sequence_length) + self.frame_dims, dtype=np.uint8)
        self.batch_norm = args.batch_norm

        self.be = gen_backend(
                backend = args.backend,
                batch_size = args.batch_size,
                rng_seed = args.random_seed,
                device_id = args.device_id,
                datatype = np.dtype(args.datatype).type,
                stochastic_round = args.stochastic_round)

        # prepare tensors once and reuse them
        self.input = self.be.empty(self.input_shape)
        self.input.lshape = self.input_shape # HACK: needed for convolutional networks
        self.targets = self.be.empty((self.output_shape, self.batch_size))

        # create model
        layers = self._create_layer()
        self.model = Model(layers = layers)
        self.cost_func = GeneralizedCost(costfunc = SumSquared())
        # Bug fix
        for l in self.model.layers.layers:
            l.parallelism = 'Disabled'
        self.model.initialize(self.input_shape[:-1], self.cost_func)

        self._set_optimizer()

        if not self.args.load_weights == None:
            self.load_weights(self.args.load_weights)

        # create target model
        if self.target_update_frequency:
            layers = self._create_layer()
            self.target_model = Model(layers)
            # Bug fix
            for l in self.target_model.layers.layers:
                l.parallelism = 'Disabled'
            self.target_model.initialize(self.input_shape[:-1])
        else:
            self.target_model = self.model

        self.callback = None
        _logger.debug("%s" % self)

    def _create_layer(self):
        """ Build a network consistent with the DeepMind Nature paper. """
        _logger.debug("Output shape = %d" % self.output_shape)
        # create network
        init_norm = Gaussian(loc=0.0, scale=0.01)
        layers = []
        # The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity.
        layers.append(
                Conv((8, 8, 32),
                strides=4,
                init=init_norm,
                activation=Rectlin(),
                batch_norm=self.batch_norm))
        # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
        layers.append(
                Conv((4, 4, 64),
                strides=2,
                init=init_norm,
                activation=Rectlin(),
                batch_norm=self.batch_norm))
        # This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
        layers.append(
                Conv((3, 3, 64),
                strides=1,
                init=init_norm,
                activation=Rectlin(),
                batch_norm=self.batch_norm))
        # The final hidden layer is fully-connected and consists of 512 rectifier units.
        layers.append(
                Affine(
                    nout=512,
                    init=init_norm,
                    activation=Rectlin(),
                    batch_norm=self.batch_norm))
        # The output layer is a fully-connected linear layer with a single output for each valid action.
        layers.append(
                Affine(
                    nout= self.output_shape,
                    init = init_norm))
        return layers

    def _set_optimizer(self):
        """ Initializes the selected optimization algorithm. """
        _logger.debug("Optimizer = %s" % str(self.args.optimizer))
        if self.args.optimizer == 'rmsprop':
            self.optimizer = RMSProp(
                    learning_rate = self.args.learning_rate,
                    decay_rate = self.args.decay_rate,
                    stochastic_round = self.args.stochastic_round)
        elif self.args.optimizer == 'adam':
            self.optimizer = Adam(
                    learning_rate = self.args.learning_rate,
                    stochastic_round = self.args.stochastic_round)
        elif self.args.optimizer == 'adadelta':
            self.optimizer = Adadelta(
                    decay = self.args.decay_rate,
                    stochastic_round = self.args.stochastic_round)
        else:
            assert false, "Unknown optimizer"

    def _prepare_network_input(self, states):
        """ Transforms and normalizes the states from one minibatch.

        Args:
            states (): a set of states with the size of minibatch
        """
        _logger.debug("Normalizing and transforming input")
        # change order of axes to match what Neon expects
        states = np.transpose(states, axes = (1, 2, 3, 0))
        # copy() shouldn't be necessary here, but Neon doesn't work otherwise
        self.input.set(states.copy())
        # normalize network input between 0 and 1
        self.be.divide(self.input, self.grayscales, self.input)

    def train(self, minibatch, epoch):
        """ Prepare, perform and document a complete train step for one minibatch.

        Args:
            minibatch (numpy.ndarray): Mini-batch of states, shape=(batch_size,sequence_length,frame_width,frame_height)
            epoch (int): Current train epoch
        """
        _logger.debug("Complete trainig step for one minibatch")
        prestates, actions, rewards, poststates, terminals = minibatch
        assert len(prestates.shape) == 4
        assert len(poststates.shape) == 4
        assert len(actions.shape) == 1
        assert len(rewards.shape) == 1
        assert len(terminals.shape) == 1
        assert prestates.shape == poststates.shape
        assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]
        # feed-forward pass for poststates to get Q-values
        self._prepare_network_input(poststates)
        postq = self.target_model.fprop(self.input, inference = True)
        assert postq.shape == (self.output_shape, self.batch_size)
        # calculate max Q-value for each poststate
        maxpostq = self.be.max(postq, axis=0).asnumpyarray()
        assert maxpostq.shape == (1, self.batch_size)
        # average maxpostq for stats
        maxpostq_avg = maxpostq.mean()
        # feed-forward pass for prestates
        self._prepare_network_input(prestates)
        preq = self.model.fprop(self.input, inference = False)
        assert preq.shape == (self.output_shape, self.batch_size)
        # make copy of prestate Q-values as targets
        targets = preq.asnumpyarray()
        # clip rewards between -1 and 1
        rewards = np.clip(rewards, self.min_reward, self.max_reward)
        # update Q-value targets for each state only at actions taken
        for i, action in enumerate(actions):
            if terminals[i]:
                targets[action, i] = float(rewards[i])
            else:
                targets[action, i] = float(rewards[i]) + self.discount_rate * maxpostq[0,i]
        # copy targets to GPU memory
        self.targets.set(targets)
        # calculate errors
        errors = self.cost_func.get_errors(preq, self.targets)
        assert errors.shape == (self.output_shape, self.batch_size)
        # average error where there is a error (should be 1 in every row)
        #TODO: errors_avg = np.sum(errors)/np.size(errors[errors>0.])
        # clip errors
        if self.clip_error:
            self.be.clip(errors, -self.clip_error, self.clip_error, out = errors)
        # calculate cost, just in case
        cost = self.cost_func.get_cost(preq, self.targets)
        assert cost.shape == (1,1)
        # perform back-propagation of gradients
        self.model.bprop(errors)
        # perform optimization
        self.optimizer.optimize(self.model.layers_to_optimize, epoch)
        # increase number of weight updates (needed for target clone interval)
        self.update_iterations += 1
        if self.target_update_frequency and self.update_iterations % self.target_update_frequency == 0:
            self._copy_theta()
            _logger.info("Network update #%d: Cost = %s, Avg Max Q-value = %s" % (self.update_iterations, str(cost.asnumpyarray()[0][0]), str(maxpostq_avg)))
        # update statistics
        if self.callback:
            self.callback.from_learner(cost.asnumpyarray()[0,0], maxpostq_avg)

    def get_Q(self, state):
        """ Calculates the Q-values for one mini-batch.

        Args:
            state(numpy.ndarray): Single state, shape=(sequence_length,frame_width,frame_height).

        Returns:
            q_values (numpy.ndarray): Results for first element of mini-batch from one forward pass through the network, shape=(self.output_shape,)
        """
        _logger.debug("State shape = %s" % str(state.shape))
        # minibatch is full size, because Neon doesn't let change the minibatch size
        # so we need to run 32 forward steps to get the one we actually want
        self.dummy_batch[0] = state
        states = self.dummy_batch
        assert states.shape == ((self.batch_size, self.sequence_length,) + self.frame_dims)
        # calculate Q-values for the states
        self._prepare_network_input(states)
        qvalues = self.model.fprop(self.input, inference = True)
        assert qvalues.shape == (self.output_shape, self.batch_size)
        _logger.debug("Qvalues: %s" % (str(qvalues.asnumpyarray()[:,0])))
        return qvalues.asnumpyarray()[:,0]

    def _copy_theta(self):
        """ Copies the weights of the current network to the target network. """
        _logger.debug("Copying weights")
        pdict = self.model.get_description(get_weights=True, keep_states=True)
        self.target_model.deserialize(pdict, load_states=True)

    def save_weights(self, target_dir, epoch):
        """ Saves the current network parameters to disk.

        Args:
            target_dir (str): Directory where the network parameters are stored for each episode.
            epoch (int): Current epoch.
        """
        filename = "%s_%s_%s_%d.prm" % (str(self.args.game.lower()), str(self.args.net_type.lower()), str(self.args.optimizer.lower()), (epoch + 1))
        self.model.save_params(os.path.join(target_dir, filename))

    def load_weights(self, source_file):
        """ Loads the network parameters from a given file.

        Args:
            source_file (str): Complete path to a file with network parameters.
        """
        self.model.load_params(source_file)
