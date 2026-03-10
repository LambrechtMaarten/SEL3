from configs.subconfiguration import SubConfiguration
from src.controller import Controller, StandardController, BodyDirectionController


class ControllerConfiguration(SubConfiguration):
    """
    This class contains the configuration determining which brittle star robot controller is used.
    """

    def __init__(self, name, controller: Controller):
        super().__init__(name)
        self.controller = controller


def standard():
    return ControllerConfiguration(
        "standard",
        StandardController()
    )


def body_direction():
    return ControllerConfiguration(
        "body_direction",
        BodyDirectionController()
    )
