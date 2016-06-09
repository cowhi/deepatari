import logging
_logger = logging.getLogger(__name__)

import abc

class Environment(object):
    """ This is the base class for all environment implementations.

    The environment class provides an interface to the environment for the agent
    and allows the agent to interact with it and receive observations of state.

    Attributes:
        name (str): The name of the network object.

    Note:
        All subclasses must inherite this class --> 'class NewClass(Environment)'

    """
    __metaclass__ = abc.ABCMeta

    def __str__(self):
        """ Overwrites the object.__str__ method.

        Returns:
            string (str): Important parameters of the object.
        """
        return "I am a " + str(type(self).__name__) + "\n" \
           "My name is " + str(self.name)

    def __init__(self, name):
        """ Initializes a given environment.

        Args:
            name (str): The name of the environment object.

        """
        _logger.info("Initialize object of type " + str(type(self).__name__))
        self.name = name

    @abc.abstractmethod
    def step(self):
        """ Individual step function for each subclass """
        pass

    @abc.abstractmethod
    def reset_env(self):
        """ Individual step function for each subclass """
        pass
