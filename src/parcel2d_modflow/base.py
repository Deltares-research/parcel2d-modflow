from abc import ABC, abstractmethod


class AbstractModule(ABC):
    """
    Abstract base class for modelling components in SOMERS. Inheritance makes sure that
    all subclasses have an initialize and run method.

    Attributes
    ----------
    available_modules : set
        Set of all available modules in SOMERS. Each module that inherits from `AbstractModule`
        is automatically added to the set. This is used to check if a module is valid.
    """

    available_modules = set()

    def __init_subclass__(cls, **kwargs):
        """
        Register any Module subclass in SOMERS that inherits from this abstract base class.
        """
        super().__init_subclass__(**kwargs)
        cls.available_modules.add(cls.__name__)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def is_valid(cls, module: str) -> bool:
        """
        Check if a module is valid Model component in SOMERS.

        Parameters
        ----------
        module : str
            Name of module to check.

        Returns
        -------
        bool
            True, if the module exists in the registry.

        """
        return module in cls.available_modules
