from configs.subconfiguration import SubConfiguration
from src.controller import Controller


class ControllerConfiguration(SubConfiguration):
    """
    This class contains the configuration determining which brittle star robot controller is used.
    """

    def __init__(self, name, controller: Controller):
        super().__init__(name)
        self.controller = controller
