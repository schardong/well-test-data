# coding: utf-8


class Error(Exception):
    """Base class for exceptions of this module. """
    pass


class AuthenticationError(Error):
    """Exception raised when the authentication fails.
    :param message: (Optional) Explanation about the error
    """
    def __init__(self, message='Authentication error.'):
        self.message = message


class ProjectCreationError(Error):
    """Exception raised when the project creation process fails.
    :param message: (Optional) Explanation about the error
    """
    def __init__(self, message='Failed to create project.'):
        self.message = message


class ProjectRemovalError(Error):
    """Exception raised when the project removal process fails.
    :param message: (Optional) Explanation about the error
    """
    def __init__(self, message='Failed to remove project.'):
        self.message = message


class SimulatorCreationError(Error):
    """Exception raised when the simulator creation process fails.
    :param message: (Optional) Explanation about the error
    """
    def __init__(self, message='Failed to add the simulator.'):
        self.message = message


class SimulatorRemovalError(Error):
    """Exception raised when the simulator removal process fails.
    :param message: (Optional) Explanation about the error
    """
    def __init__(self, message='Failed to remove the simulator.'):
        self.message = message


class SimulatorNotFoundError(Error):
    """Exception raised when the simulator removal process fails.
    :param message: (Optional) Explanation about the error
    """
    def __init__(self, message='Requested simulator not found.'):
        self.message = message


class MissingFieldError(Error):
    """Exception raised when a data field is missing while deserializing
    an object"""
    def __init__(self, message='Missing field while deserializing object.'):
        self.message = message


class MissingBaseSimulationFileError(Error):
    """Exception raised when the base simulation file is missing."""
    def __init__(self, message='Missing base simulation file.'):
        self.message = message
