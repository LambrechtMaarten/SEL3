from abc import ABC


class SubConfiguration(ABC):
    def __init__(self, name):
        self.name = name
        self.configuration = None

    def set_configuration(self, config):
        self.configuration = config
