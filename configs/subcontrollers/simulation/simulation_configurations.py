from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification
from moojoco.mjcf.arena import ArenaConfiguration

from configs.subconfiguration import SubConfiguration
from src.numerical_analysis.numerical_analysis import DifferentialEquationSolver


class SimulationConfiguration(SubConfiguration):
    """
    This class configures all details of the simulation.
    This includes the arena and morphology, as well as the simulation time and step size.
    """

    def __init__(
            self,
            name: str,
            solver: DifferentialEquationSolver,
            environment_configuration: BrittleStarUndirectedLocomotionEnvironmentConfiguration,
            arena_configuration: ArenaConfiguration,
            morphology_configuration: BrittleStarMorphologySpecification):
        super().__init__(name)
        self.solver = solver
        self.environment_configuration = environment_configuration
        self.arena_configuration = arena_configuration
        self.morphology_configuration = morphology_configuration