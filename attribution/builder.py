"""
Builder object used to construct various 
"""
import abc

class BaseBuilder(abc.ABC):
  """
  We'll follow a builder pattern here to construct objects
  Attributes:

    handlers (dict): key-value pairs mapping keys to specifi
                     subclasses (e.g., DatabaseLocation)

  Methods:
    register: Register a new subclass (handler)
    get_handler: Get handler (value) corresponding to a key
  """

  def __init__(self, *largs, **kwargs):
    super(abc.ABC, self).__init__()

    self.handlers = dict()

  def register(self, handler_key, handler):
    """
    Registers a new subclass in handlers dictionary.
    Args:
      handler_key (tuple): dictionary key
      handler (class): subclass
    """

    self.handlers[handler_key] = handler

  def get_handler(self, handler_key):
    """
    Find subclass based on key.

    Args:
      handler_key (tuple):

    Returns:
    """

    return self.handlers[handler_key]

  @abc.abstractmethod
  def build(self):
    """
    This is the custom piece of a builder.

    It will returned a constructed object
    """

    raise NotImplementedError()
