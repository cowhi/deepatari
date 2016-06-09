import numpy as np
import logging
_logger = logging.getLogger(__name__)

class AtariStateBuffer(object):
    """ A buffer to collect frames until they form a state.

    Attributes:
        name (str): The name of the buffer object.
        sequence_length (int): Determines how many frames form a state.
        frame_dims (tuple[int]): Dimension of the resized frames.
        buffer (numpy.ndarray): Actual buffer.
        buffer_size (tuple[int]): Size of the buffer.

    """


    def __str__(self):
        """ Overwrites the object.__str__ method.

        Returns:
            string (str): Important parameters of the object.
        """
        return "'name':" + str(self.name) + ", " + \
               "'sequence_length':" + str(self.sequence_length) + ", " + \
               "'frame_dims':" + str(self.frame_dims ) + ", " + \
               "'buffer_size':" + str(self.buffer_size)

    def __init__(self, args, name = "AtariStateBuffer"):
        """ Initializes a buffer object to collect observations until they qualify as state.

        Args:
            args (argparse.Namespace): All settings either default or set via command line arguments.
            name (str): The name of the StateBuffer object.
        """
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        self.name = name
        self.sequence_length = args.sequence_length
        self.frame_dims = (args.frame_height, args.frame_width)
        self.buffer = np.zeros((self.sequence_length, self.frame_dims[0], self.frame_dims[1]), dtype=np.uint8)
        self.buffer_size = np.shape(self.buffer)
        _logger.debug("%s" % self)

    def add(self, frame):
        """ Adds a frame to the buffer.

        Args:
            frame (tuple[int]): The current game frame after the last action.
        """
        _logger.debug("Buffer shape = %s)" % str(self.buffer.shape))
        assert frame.shape == self.frame_dims
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = frame

    def get_current_state(self):
        """ Returns a full state representation from the buffer.

        Returns:
            buffer (numpy.ndarray): The current StateBuffer.
        """
        _logger.debug("Buffer shape = %s" % str(self.buffer.shape))
        return self.buffer

    def reset(self):
        """ Resets the buffer to all zeros. """
        _logger.debug("All zeros")
        self.buffer *= 0
