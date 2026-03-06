from abc import ABC


class SubConfiguration(ABC):
    def __init__(self, name):
        self.name = name
        self.configuration = None
        self.children = []

    def set_configuration(self, config):
        self.configuration = config
        for child in self.children:
            child.set_configuration(config)
